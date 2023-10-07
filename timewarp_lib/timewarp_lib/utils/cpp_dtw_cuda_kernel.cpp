#include <torch/extension.h>

#include <iostream>

#define VERBOSE false

// Forward-define CUDA
void get_dtw_path(torch::Tensor distance_matrix_base, torch::Tensor& path_base, torch::Tensor& warp_matrix_base);

torch::Tensor dtw_warp_first_to_second(torch::Tensor recon, torch::Tensor actual) {
  int64_t batch_size = actual.size(0);
  int64_t actual_time_steps = actual.size(1);
  int64_t num_channels = actual.size(2);
  int64_t recon_time_steps = recon.size(1);

  // allocate max size possible torch matrix
  // because you can't use std::vector on CUDA and
  // "maybe cuda will be smart" or something...
  // then fill in the path_base array
  auto first_reshaped = recon.reshape({batch_size, recon_time_steps, 1, num_channels});
  auto second_reshaped = actual.reshape({batch_size, 1, actual_time_steps, num_channels});
  auto distance_matrix_base = torch::sum(torch::square(first_reshaped - second_reshaped),/*dim*/3);

  auto path_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto warp_options = torch::TensorOptions().device(torch::kCUDA);

  auto path_base = torch::zeros({batch_size,recon_time_steps+actual_time_steps+1,2},path_options);
  auto warp_matrix_base = torch::zeros({batch_size, actual_time_steps, recon_time_steps},warp_options);
  get_dtw_path(distance_matrix_base, path_base, warp_matrix_base);

  torch::Tensor warped_recon = torch::einsum("bar,brc->bac", {warp_matrix_base, recon});

  return warped_recon;
}

torch::Tensor dtw_loss(torch::Tensor recon, torch::Tensor actual) {
  torch::Tensor warped_recon = dtw_warp_first_to_second(recon, actual);
  int64_t num_channels = actual.size(2);
  auto loss = torch::nn::functional::mse_loss(warped_recon, actual, torch::enumtype::kMean()) * num_channels;
  return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dtw_loss", &dtw_loss, "dtw loss in cpp");
  m.def("dtw_warp_first_to_second", &dtw_warp_first_to_second, "dtw warp first to second");
}
