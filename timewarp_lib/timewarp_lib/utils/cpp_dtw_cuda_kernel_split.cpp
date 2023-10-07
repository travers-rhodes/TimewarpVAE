#include <torch/extension.h>

#include <iostream>

#define VERBOSE false

// Forward-define CUDA
void fill_parent_path_dict(torch::Tensor distance_matrix_base, torch::Tensor& parent_path_dictionary_base);
void fill_warp_recon_to_actual(torch::Tensor parent_path_dictionary_base, torch::Tensor& warp_matrix_base);
void fill_warp_recon_and_actual(torch::Tensor parent_path_dictionary_base, torch::Tensor& warp_matrix_first_base, torch::Tensor& warp_matrix_second_base);

std::vector<torch::Tensor> dtw_warp_first_and_second(torch::Tensor recon, torch::Tensor actual) {
  int64_t batch_size = actual.size(0);
  int64_t actual_time_steps = actual.size(1);
  int64_t num_channels = actual.size(2);
  int64_t recon_time_steps = recon.size(1);
  
  auto path_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto parent_path_dictionary_base = torch::zeros({batch_size, recon_time_steps, actual_time_steps},path_options);

  // apparently local scope is a good way to clear memory?
  // https://discuss.pytorch.org/t/how-to-manually-delete-free-a-tensor-in-aten/64153/4
  {
    // allocate max size possible torch matrix
    // because you can't use std::vector on CUDA and
    // "maybe cuda will be smart" or something...
    // then fill in the path_base array
    auto first_reshaped = recon.reshape({batch_size, recon_time_steps, 1, num_channels});
    auto second_reshaped = actual.reshape({batch_size, 1, actual_time_steps, num_channels});
    auto distance_matrix_base = torch::sum(torch::square(first_reshaped - second_reshaped),/*dim*/3);

    fill_parent_path_dict(distance_matrix_base, parent_path_dictionary_base);
  }

  int64_t maxlen = actual_time_steps + recon_time_steps + 1;
  auto warp_options = torch::TensorOptions().device(torch::kCUDA);
  auto warp_matrix_first_base = torch::zeros({batch_size, maxlen, recon_time_steps},warp_options);
  auto warp_matrix_second_base = torch::zeros({batch_size, maxlen, actual_time_steps},warp_options);
  fill_warp_recon_and_actual(parent_path_dictionary_base, warp_matrix_first_base, warp_matrix_second_base);
  auto warped_first = torch::einsum("bor,brc->boc", {warp_matrix_first_base, recon});
  auto warped_second = torch::einsum("boa,bac->boc", {warp_matrix_second_base, actual});
  //https://discuss.pytorch.org/t/how-to-get-multiple-tensors-returned-from-cuda-extension/52524
  std::vector<torch::Tensor> outputs;
  outputs.push_back(warped_first);
  outputs.push_back(warped_second);
  return outputs;
}

torch::Tensor dtw_warp_first_to_second(torch::Tensor recon, torch::Tensor actual) {
  int64_t batch_size = actual.size(0);
  int64_t actual_time_steps = actual.size(1);
  int64_t num_channels = actual.size(2);
  int64_t recon_time_steps = recon.size(1);
  
  auto path_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto parent_path_dictionary_base = torch::zeros({batch_size, recon_time_steps, actual_time_steps},path_options);

  // apparently local scope is a good way to clear memory?
  // https://discuss.pytorch.org/t/how-to-manually-delete-free-a-tensor-in-aten/64153/4
  {
    // allocate max size possible torch matrix
    // because you can't use std::vector on CUDA and
    // "maybe cuda will be smart" or something...
    // then fill in the path_base array
    auto first_reshaped = recon.reshape({batch_size, recon_time_steps, 1, num_channels});
    auto second_reshaped = actual.reshape({batch_size, 1, actual_time_steps, num_channels});
    auto distance_matrix_base = torch::sum(torch::square(first_reshaped - second_reshaped),/*dim*/3);

    fill_parent_path_dict(distance_matrix_base, parent_path_dictionary_base);
  }

  auto warp_options = torch::TensorOptions().device(torch::kCUDA);
  auto warp_matrix_base = torch::zeros({batch_size, actual_time_steps, recon_time_steps},warp_options);
  fill_warp_recon_to_actual(parent_path_dictionary_base, warp_matrix_base);

  auto warped_recon = torch::einsum("bar,brc->bac", {warp_matrix_base, recon});
  return warped_recon;
}

torch::Tensor dtw_loss(torch::Tensor recon, torch::Tensor actual) {
  int64_t num_channels = actual.size(2);
  auto warped_recon = dtw_warp_first_to_second(recon,actual);
  auto loss = torch::nn::functional::mse_loss(warped_recon, actual, torch::enumtype::kMean()) * num_channels;
  return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dtw_loss", &dtw_loss, "dtw loss in cpp");
  m.def("dtw_warp_first_to_second", &dtw_warp_first_to_second, "dtw warp first to second");
  m.def("dtw_warp_first_and_second", &dtw_warp_first_and_second, "dtw warp first and second");
}
