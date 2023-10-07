#include <torch/extension.h>

__global__
void basic_dtw_cuda_kernel(const torch::PackedTensorAccessor32<float,3> distance_matrix,
                   torch::PackedTensorAccessor32<int,3> parent_path_dictionary,
                   torch::PackedTensorAccessor32<float,3> warping_cost,
                   torch::PackedTensorAccessor32<int,3> path,
                   torch::PackedTensorAccessor32<float,3> warp_matrix) {
  int64_t recon_time_steps = warp_matrix.size(1);
  int64_t actual_time_steps = warp_matrix.size(2);
  int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_id < distance_matrix.size(0)){

    // for computational simplicity, we assume
    // path costs are 1 for left, diag, and right
    int64_t first_ts = distance_matrix.size(1);
    int64_t second_ts = distance_matrix.size(2);

    // Taking as input
    // ***distance_matrix***
    // Use Dynamic Programming to compute the
    // ***warping_cost*** of a path going to the relevant cell.
    // For ease of subsequent calculation, also keep track
    // of how that cell was reached (eg: diagonally or horizontally)
    // Result is coded in 
    // ***parent_path_dictionary*** using 0,1,2 for
    // parent equal to [(i-1,j-1),(i-1, j),(i][j-1)]
    for (int64_t i = 0; i < first_ts; i++) {
      for (int64_t j = 0; j < second_ts; j++) {
        if (i==0 and j==0) {
          warping_cost[batch_id][i][j] = 0;
        } else if (i == 0) {
          parent_path_dictionary[batch_id][i][j] = 2;
          warping_cost[batch_id][i][j] = warping_cost[batch_id][i][j-1] + distance_matrix[batch_id][i][j];
        } else if (j == 0) {
          parent_path_dictionary[batch_id][i][j] = 1;
          warping_cost[batch_id][i][j] = warping_cost[batch_id][i-1][j] + distance_matrix[batch_id][i][j];
        } 
          // this next part only looks so clean because
          // we're assuming path costs of 1 for left, diag, right.
          // the ordering is to prefer diag over the others if all equal.
          else if (((warping_cost[batch_id][i-1][j-1] <= warping_cost[batch_id][i][j-1]) &&
                    (warping_cost[batch_id][i-1][j-1] <= warping_cost[batch_id][i-1][j]))) {
          parent_path_dictionary[batch_id][i][j] = 0;
          warping_cost[batch_id][i][j] = warping_cost[batch_id][i-1][j-1] + distance_matrix[batch_id][i][j];
        } else if ((warping_cost[batch_id][i-1][j] <= warping_cost[batch_id][i][j-1])) {
          parent_path_dictionary[batch_id][i][j] = 1;
          warping_cost[batch_id][i][j] = warping_cost[batch_id][i-1][j] + distance_matrix[batch_id][i][j];
        } else {
          parent_path_dictionary[batch_id][i][j] = 2;
          warping_cost[batch_id][i][j] = warping_cost[batch_id][i][j-1] + distance_matrix[batch_id][i][j];
        }
      }
    }

    // using as input the
    // ***parent_path_dictionary***
    // fill in 
    // ***path***
    // backward from end to beginning
    // path ends by definition when it hits (0,0)
    // so, default fill with 0,0s
    int64_t path_len = 0;
    int64_t cur_cell_x = first_ts-1;
    int64_t cur_cell_y = second_ts-1;
    path[batch_id][path_len][0] = cur_cell_x;
    path[batch_id][path_len][1] = cur_cell_y;
    path_len++;
    while (cur_cell_x != 0 or cur_cell_y != 0) {
      int64_t parent_direction = parent_path_dictionary[batch_id][cur_cell_x][cur_cell_y];
      if (parent_direction == 0) {
        cur_cell_x--;
        cur_cell_y--;
      } else if (parent_direction == 1) {
        cur_cell_x--;
      } else {
        cur_cell_y--;
      }

      path[batch_id][path_len][0] = cur_cell_x;
      path[batch_id][path_len][1] = cur_cell_y;
      path_len++;
    }

    // using just the
    // ***path***
    // compute the
    // ***warp_matrix***
    // go through the whole path once, keeping track of when we start
    // and stop matching a particular actual_index
    // since we go through the path backward (from the last to first)
    // at the beginning, the current_actual_index is the last index
    int64_t current_actual_index = actual_time_steps-1;
    int64_t start_matching_index = 0;
    int64_t path_index = 0;
    // pair_inds are (recon, actual)
    auto pair_inds = path[batch_id][path_index];
    // (0,0) means you're on your last loop
    bool already_reached_end_of_path = false;
    while (not already_reached_end_of_path) {
      pair_inds = path[batch_id][path_index];
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
          auto copy_pair = path[batch_id][copy_index];
          // warp_matrix indices are "bar". Not actually smart/useful, 
          // but note how it's different from path_index which is "(r,a)"
          warp_matrix[batch_id][copy_pair[1]][copy_pair[0]] =  1./num_matching;
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
          auto copy_pair = path[batch_id][copy_index];
          // warp_matrix indices are "bar". Not actually smart/useful, 
          // but note how it's different from path_index which is "(r,a)"
          warp_matrix[batch_id][copy_pair[1]][copy_pair[0]] =  1./num_matching;
        }
        start_matching_index = path_index;
        current_actual_index = pair_inds[1];
      }
      path_index++;
      already_reached_end_of_path = now_at_end_of_path;
    }
  }
}

// given a matrix of shape (batch_size, recon_time_steps, actual_time_steps)
// giving the distance between recon and actual
// compute the dtw path that aligns the recon and actual
// with two outputs:
// path_base of shape (batch_size, recon_time_steps + actual_time_steps + 1, 2)
// giving pairs (recon_time, actual_time) along the path
// path_base is an "out" parameter that gets filled in
// and also
// warp_matrix
// compute the warping matrix you can use to convert the recon data to align it
// to the actual data
// warp_matrix_base is an "out" parameter that gets filled in
void get_dtw_path(torch::Tensor distance_matrix_base, torch::Tensor& path_base, torch::Tensor& warp_matrix_base) {
  auto device = distance_matrix_base.device();
  int64_t batch_size = distance_matrix_base.size(0);
  int64_t recon_time_steps = distance_matrix_base.size(1);
  int64_t actual_time_steps = distance_matrix_base.size(2);
  
  auto path_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto warp_options = torch::TensorOptions().device(torch::kCUDA);

  auto parent_path_dictionary_base = torch::zeros_like(distance_matrix_base,path_options);
  auto warping_cost_base = torch::zeros_like(distance_matrix_base,warp_options);
  
  auto parent_path_dictionary = parent_path_dictionary_base.packed_accessor32<int,3>();
  auto warping_cost = warping_cost_base.packed_accessor32<float,3>();
  auto distance_matrix = distance_matrix_base.packed_accessor32<float,3>();
  auto path = path_base.packed_accessor32<int,3>();
  auto warp_matrix = warp_matrix_base.packed_accessor32<float,3>();

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(distance_matrix_base.type(), "basic_dtw_cuda", ([&] {
    basic_dtw_cuda_kernel<<<blocks,threads>>>(distance_matrix, parent_path_dictionary, warping_cost, path, warp_matrix);
  }));
}
