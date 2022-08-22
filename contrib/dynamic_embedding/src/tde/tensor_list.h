#pragma once
#include <torch/torch.h>

namespace tde {

class TensorList: public torch::CustomClassHolder {
 public:
  TensorList() = default;

  void push_back(at::Tensor tensor) { tensors_.push_back(tensor); }
  int64_t size() const { return tensors_.size(); }
  torch::Tensor operator[](int64_t index) { return tensors_[index]; }

 private:
  std::vector<torch::Tensor> tensors_;
};

} // namespace tde
