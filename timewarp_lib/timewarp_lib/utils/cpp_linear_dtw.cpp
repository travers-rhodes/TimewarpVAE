#include <torch/extension.h>

#include <iostream>

#define VERBOSE false

//
// The linear DTW algorithm shall work as follows:
// Preconditions:
// 1) One time dimension shall be much larger than the other
// 2) Path segments shall take exactly one step in the smaller time interval
//    and at least one step in the larger time interval.
// 3) The final path cost shall be a sum of path segment costs
//    (pairwise mean square error times the number of larger time interval steps)
//    plus the ``slope regularization''
//    which is a function of just:
//       a) the total number of larger time interval steps
//       b) the total number of smaller time interval steps
//       c) the current number of larger time interval steps
// Result:
// 1) The returned path shall (be one of the paths that) have the smallest cost
// Algorithm complexity bounds:
//   The GPU memory usage shall not be more than batchsize x longer time length squared x a constant
//   The GPU kernel shall be called no more than shorter time length x a constant times
//   (Linearly interpolated) distance calculations can be re-computed repeatedly
// Later, we can try to worry about beating those bounds


// the regularization function encourages
// steps to have unit slope (after subsegmenting so that short_length is
// made as granular as long_length)
// TODO: Is it ``weird" that we integrate over d_reconstruction (not symmetric)?
float regularization_function(int long_steps, int long_length, int short_length) {
  float slope = long_steps/(float(long_length)/ short_length);
  float logslope = log(slope);
  float regularization = logslope * logslope;
  return regularization;
}

// the distance function is computed as follows.
// first, interpolate short dataset so it matches the granularity
// traversed by the long dataset segment
// then, match up and return the sum squared distance (excluding the first point, 
// and including the last point)
// [long_start_0, long_1,         long_2,         long_3,         long_end_4]
//                   |              |               |                  |
// [short_start,  short_interp_1, short_interp_2, short_interp_3, short_end]
// graphically, compare and add the difference between the values above.
// note that long_start and short_start were already compared at the previous segment.
float distance_function(int long_start, int long_end, int short_start, int short_end, torch::Tensor long_mat_base, torch::Tensor short_mat_base) {
  auto short_mat = short_mat_base.accessor<float,2>();
  auto long_mat = long_mat_base.accessor<float,2>();
  int measurement_dims = short_mat.size(1);
  // compute the number of steps between long_start and long_end
  auto intervals = long_end - long_start;
  // linearly interpolate short data that many timesteps
  // compute the distance during interpolation, so that we don't have more than one point in memory
  // at a time
  auto start = short_mat[short_start];
  auto end = short_mat[short_end];
  float distance = 0.0;
  for (int i = 1; i <= intervals; i++) {
    // because tensor/vector operations are expensive (and/or confusing) in cpp
    // we manually loop through the final dimension of the vectors and sum the squared errors
    for (int dimind = 0; dimind < measurement_dims; dimind++) {
      auto interpolation_i_dimind = start[dimind] + (float(i)/intervals) * (end[dimind]-start[dimind]);
      auto dimdist = interpolation_i_dimind - long_mat[long_start+i][dimind];
      distance += dimdist * dimdist;
    }
  }
  return(distance);
}

torch::Tensor get_dtw_path(torch::Tensor first, torch::Tensor second, float regularization_lambda) {
  int first_ts = first.size(0);
  int second_ts = second.size(0);

  auto warping_cost_base = torch::full({first_ts,second_ts}, INFINITY).to(first.device());
  auto warping_cost = warping_cost_base.accessor<float,2>();

  // look up using indexing. Result i indices stored in tensor.
  // j index is always just j-1 (always exactly one step in smaller dim)
  auto path_options = torch::TensorOptions().dtype(torch::kInt32);
  auto parent_path_dictionary_base_i = torch::zeros({first_ts, second_ts}, path_options); 
  auto parent_path_dictionary_i = parent_path_dictionary_base_i.accessor<int,2>(); 
  for (int j = 0; j < second_ts; j++) {
    // per the assumptions, you can't take a zero-derivative step.
    // that is, every step needs a slope of at least one
    // or, in other words, the first timestep of each pair
    // and the next step must have one step of small direction
    // and at least one step of large direction.
    if (j==0) {
     warping_cost[0][j] = float(0.);
     for (int i = 1; i < first_ts; i++) {
       warping_cost[i][j] = INFINITY;
     }
     continue;
    }
    for (int i = 0; i < first_ts; i++) {
      float best_cost = INFINITY;
      int best_prev_i = 0;
      // compute the cheapest way to warp so that i and j timesteps align
      // the cost will (of course) be
      // the minimum of [a sum of the cost to get j-1 matched with prev_i plus cost to get from there to here]
      // with the constraint that prev_i must be at most i-1
      for (int prev_i = 0; prev_i < i; prev_i++) {
        // compute cost to get from (prev_i,j-1) to (i,j)
        float dist = distance_function(prev_i, i, j-1, j, first, second);
        float regularization = regularization_function(i-prev_i, first_ts, second_ts);
        float current_option = dist + regularization * regularization_lambda + warping_cost[prev_i][j-1];
        if (current_option < best_cost) {
          best_cost = current_option;
          best_prev_i = prev_i;
        }
      }
      warping_cost[i][j] = best_cost;
      parent_path_dictionary_i[i][j] = best_prev_i;
    }
  }

  if (VERBOSE) {
    std::cout << "Warping cost:\n";
    for (int i = 0; i < first_ts; i++) {
      for (int j = 0; j < second_ts; j++) {
        std::cout << warping_cost_base[i][j].to("cpu").data_ptr<float>()[0] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n\n";
  }
  
  if (VERBOSE) {
    std::cout << "parent path dictionary i:\n";
    for (int i = 0; i < first_ts; i++) {
      for (int j = 0; j < second_ts; j++) {
        std::cout << parent_path_dictionary_base_i[i][j].to("cpu").data_ptr<int>()[0] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n\n";
  }

  // now, path is always exactly size secont_ts + 1
  auto path_base = torch::zeros({second_ts+1,2},torch::kInt32);
  auto path = path_base.accessor<int,2>();
  int path_len = 0;
  int cur_cell_x = first_ts-1;
  int cur_cell_y = second_ts-1;
  path[path_len][0] = cur_cell_x;
  path[path_len][1] = cur_cell_y;
  path_len++;
  while (cur_cell_x != 0 or cur_cell_y != 0) {
    cur_cell_x = parent_path_dictionary_i[cur_cell_x][cur_cell_y];
    cur_cell_y = cur_cell_y-1;
    path[path_len][0] = cur_cell_x;
    path[path_len][1] = cur_cell_y;
    path_len++;
  }
  return(path_base);
}

torch::Tensor dtw_warp_first_to_second(torch::Tensor recon, torch::Tensor actual, double regularization_lambda) {
  int batch_size = actual.size(0);
  int actual_time_steps = actual.size(1);
  int num_channels = actual.size(2);
  int recon_time_steps = recon.size(1);
  auto warp_matrix_base = torch::zeros({batch_size, actual_time_steps, recon_time_steps}).to(recon.device());
  auto warp_matrix = warp_matrix_base.accessor<float,3>();


  for (int i = 0; i < batch_size; i++) {
    auto path_base = get_dtw_path(actual[i], recon[i], regularization_lambda);
    auto path = path_base.accessor<int,2>();

    if (VERBOSE) {
      std::cout << "Path: \n";
      for (int i= 0; i < recon_time_steps; i++) {
        std::cout << path[i][0] << ", " << path[i][1] << " \n";
      }
    }

    // we loop through the actual index and fill in one or two associated (recon indexed) columns
    // based on the proportion of the path completed at that actual_ind
    for (int actual_ind = 0; actual_ind < actual_time_steps; actual_ind++) {
      // note that path is inverse-ordered, so the end of this path segment
      // is the last index where the path array is bigger than the search index
      // find end of path:
      int path_end_ind = 0;
      while ((path_end_ind < recon_time_steps) && (path[path_end_ind][0] >= actual_ind)) {
        path_end_ind++;
      }
      path_end_ind--;
      if (actual_ind == path[path_end_ind][0]) {
        auto recon_ind = path[path_end_ind][1];
        warp_matrix[i][actual_ind][recon_ind] = float(1.);
      } else {
        auto path_start_ind = path_end_ind + 1;
        auto actual_start_ind = path[path_start_ind][0];
        auto actual_end_ind = path[path_end_ind][0];
        auto recon_start_ind = path[path_start_ind][1];
        auto recon_end_ind = path[path_end_ind][1];
        float fracbelow = float(actual_end_ind - actual_ind)/(actual_end_ind-actual_start_ind);
        float fracabove = 1 - fracbelow;
        warp_matrix[i][actual_ind][recon_start_ind] = fracbelow;
        warp_matrix[i][actual_ind][recon_end_ind] = fracabove;
      }
    }
  }
  if (VERBOSE) {
    std::cout << "Warp matrix:\n";
    for (int i = 0; i < batch_size; i++) {
      for (int ats = 0; ats < actual_time_steps; ats++) {
        for (int rts = 0; rts < recon_time_steps; rts++) {
          std::cout << warp_matrix_base[i][ats][rts].to("cpu").data_ptr<float>()[0] << ", ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n\n";
    }
  }

  auto warped_recon = torch::einsum("bar,brc->bac", {warp_matrix_base, recon});

  if (VERBOSE) {
    std::cout << "Warped recon:\n";
    for (int i = 0; i < batch_size; i++) {
      for (int ats = 0; ats < actual_time_steps; ats++) {
        for (int c = 0; c < num_channels; c++) {
          std::cout << warped_recon[i][ats][c].to("cpu").data_ptr<float>()[0] << ", ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n\n";
    }
  }
  return warped_recon;
}

torch::Tensor dtw_loss(torch::Tensor recon, torch::Tensor actual, double regularization_lambda) {
  auto warped_recon = dtw_warp_first_to_second(recon, actual, regularization_lambda);
  int num_channels = actual.size(2);
  auto loss = torch::nn::functional::mse_loss(warped_recon, actual, torch::enumtype::kMean()) * num_channels;
  return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dtw_loss", &dtw_loss, "dtw loss in cpp");
  m.def("dtw_warp_first_to_second", &dtw_warp_first_to_second, "dtw warp first to second");
}
