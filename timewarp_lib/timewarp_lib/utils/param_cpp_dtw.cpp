#include <torch/extension.h>

#include <iostream>

#define VERBOSE false

torch::Tensor get_dtw_path(torch::Tensor first, torch::Tensor second, float diag_cost) {
  // for computational simplicity, we assume
  // path costs are 1 for left, diag, and right
  int64_t first_ts = first.size(0);
  int64_t second_ts = second.size(0);
  int64_t num_channels = first.size(1);

  auto first_reshaped = first.reshape({first_ts, 1, num_channels});
  auto second_reshaped = second.reshape({1, second_ts, num_channels});

  auto distance_matrix_base = torch::sum(torch::square(first_reshaped - second_reshaped),/*dim*/2);
  // distance_matrix follows "ra" indices
  auto distance_matrix = distance_matrix_base.accessor<float,2>();

  // warping_cost_base follows "ra" indices
  auto warping_cost_base = torch::zeros({first_ts,second_ts});
  auto warping_cost = warping_cost_base.accessor<float,2>();

  // look up using indexing. Result is coded of 0,1,2 for
  // parent equal to [(i-1,j-1),(i-1, j),(i][j-1)]
  auto parent_path_dictionary_base = torch::empty({first_ts, second_ts}); 
  auto parent_path_dictionary = parent_path_dictionary_base.accessor<float,2>(); 
  for (int64_t i = 0; i < first_ts; i++) {
    for (int64_t j = 0; j < second_ts; j++) {
      if (i==0 and j==0) {
        warping_cost[i][j] =  0;
      } else if (i == 0) {
        parent_path_dictionary[i][j] = 2;
        warping_cost[i][j] = warping_cost[i][j-1] + distance_matrix[i][j];
      } else if (j == 0) {
        parent_path_dictionary[i][j] = 1;
        warping_cost[i][j] = warping_cost[i-1][j] + distance_matrix[i][j];
      } else {
        float diag_option = warping_cost[i-1][j-1] + diag_cost * distance_matrix[i][j];
        float vert_option = warping_cost[i-1][j] + distance_matrix[i][j];
        float horiz_option = warping_cost[i][j-1] + distance_matrix[i][j];
        if ((diag_option <= vert_option) &&
                    (diag_option <= horiz_option)) {
          parent_path_dictionary[i][j] = 0;
          warping_cost[i][j] = diag_option;
        } else if ((vert_option <= horiz_option)) {
          parent_path_dictionary[i][j] = 1;
          warping_cost[i][j] = vert_option;
        } else {
          parent_path_dictionary[i][j] = 2;
          warping_cost[i][j] = horiz_option;
        }
      }
    }
  }
  // allocate max size possible torch matrix
  // because you can't use std::vector on CUDA and
  // "maybe cuda will be smart" or something...
  // fill in path backward from end to beginning
  // path ends by definition when it hits (0,0)
  // so, default fill with 0,0s
  // path elements are (recon,actual)
  auto path_base = torch::zeros({first_ts+second_ts+1,2},torch::kInt32);
  auto path = path_base.accessor<int,2>();
  int64_t path_len = 0;
  int64_t cur_cell_x = first_ts-1;
  int64_t cur_cell_y = second_ts-1;
  path[path_len][0] = cur_cell_x;
  path[path_len][1] = cur_cell_y;
  path_len++;
  while (cur_cell_x != 0 or cur_cell_y != 0) {
    int64_t parent_direction = parent_path_dictionary[cur_cell_x][cur_cell_y];
    if (parent_direction == 0) {
      cur_cell_x--;
      cur_cell_y--;
    } else if (parent_direction == 1) {
      cur_cell_x--;
    } else {
      cur_cell_y--;
    }

    path[path_len][0] = cur_cell_x;
    path[path_len][1] = cur_cell_y;
    path_len++;
  }
  return(path_base);
}

torch::Tensor dtw_warp_first_to_second(torch::Tensor recon, torch::Tensor actual, float diag_cost) {
  int64_t batch_size = actual.size(0);
  int64_t actual_time_steps = actual.size(1);
  int64_t recon_time_steps = recon.size(1);
  auto warp_matrix_base = torch::zeros({batch_size, actual_time_steps, recon_time_steps}).to(recon.device());
  auto warp_matrix = warp_matrix_base.accessor<float,3>();


  for (int64_t i = 0; i < batch_size; i++) {
    auto path_base = get_dtw_path(recon[i], actual[i], diag_cost);
    // path elements are (recon,actual)
    auto path = path_base.accessor<int,2>();
    
    if (VERBOSE) {
      std::cout << "Path: \n";
      for (int64_t i= 0; i < actual_time_steps + recon_time_steps + 1; i++) {
        std::cout << path[i][0] << ", " << path[i][1] << " \n";
      }
    }

    // go through the whole path once, keeping track of when we start
    // and stop matching a particular actual_index
    // since we go through the path backward (from the last to first)
    // at the beginning, the current_actual_index is the last index
    int64_t current_actual_index = actual_time_steps-1;
    int64_t start_matching_index = 0;
    int64_t path_index = 0;
    // pair_inds are (recon, actual)
    auto pair_inds = path[path_index];
    // (0,0) means you're on your last loop
    bool already_reached_end_of_path = false;
    while (not already_reached_end_of_path) {
      pair_inds = path[path_index];
      bool now_at_end_of_path = (pair_inds[0] == 0) && (pair_inds[1] == 0);

      if (pair_inds[1] != current_actual_index) {
        // we've stopped matching the previous match
        // so fill out the previous match (regardless of whether the current
        // cell is the last cell)
        int64_t last_matching_index = path_index - 1;
        int64_t num_matching = last_matching_index - start_matching_index + 1;
        for (int64_t copy_index = start_matching_index ;
             copy_index <= last_matching_index;
             copy_index++) {
          auto copy_pair = path[copy_index];
          // warp_matrix indices are "bar". Not actually smart/useful, 
          // but note how it's different from path_index which is "(r,a)"
          warp_matrix[i][copy_pair[1]][copy_pair[0]] =  1./num_matching;
        }
        start_matching_index = path_index;
        current_actual_index = pair_inds[1];
      }
      if (now_at_end_of_path) {
        // The current index is the last correct match if we're now_at_end_of_path
        // Otherwise, the previous index was the last correct match 
        int64_t last_matching_index =  path_index;
        int64_t num_matching = last_matching_index - start_matching_index + 1;
        for (int64_t copy_index = start_matching_index ;
             copy_index <= last_matching_index;
             copy_index++) {
          auto copy_pair = path[copy_index];
          // warp_matrix indices are "bar". Not actually smart/useful, 
          // but note how it's different from path_index which is "(r,a)"
          warp_matrix[i][copy_pair[1]][copy_pair[0]] =  1./num_matching;
        }
        start_matching_index = path_index;
        current_actual_index = pair_inds[1];
      }
      path_index++;
      already_reached_end_of_path = now_at_end_of_path;
    }
  }

  if (VERBOSE) {
    std::cout << "Warp matrix:\n";
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t ats = 0; ats < actual_time_steps; ats++) {
        for (int64_t rts = 0; rts < recon_time_steps; rts++) {
          std::cout << warp_matrix_base[i][ats][rts].to("cpu").data_ptr<float>()[0] << ", ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n\n";
    }
  }

  auto warped_recon = torch::einsum("bar,brc->bac", {warp_matrix_base, recon});
  return warped_recon;
}

torch::Tensor dtw_loss(torch::Tensor recon, torch::Tensor actual, float diag_cost) {
  int64_t num_channels = actual.size(2);
  auto warped_recon = dtw_warp_first_to_second(recon,actual,diag_cost);
  auto loss = torch::nn::functional::mse_loss(warped_recon, actual, torch::enumtype::kMean()) * num_channels;
  return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dtw_loss", &dtw_loss, "dtw loss in cpp");
  m.def("dtw_warp_first_to_second", &dtw_warp_first_to_second, "dtw warp first to second");
}
