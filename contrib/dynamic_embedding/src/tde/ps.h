#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/io.h"

namespace tde {

struct LocalShard {
  int64_t row_start_;
  int64_t col_start_;
  int64_t row_size_;
  int64_t col_size_;
  torch::Tensor tensor_;

  torch::Tensor GetTensorView(int64_t global_id) {
    if (global_id < row_start_ || global_id >= row_start_ + row_size_) {
      return torch::Tensor{};
    }
    return tensor_.slice(0, global_id - row_start_, global_id - row_start_ + 1);
  }
};

class LocalShardList : public torch::CustomClassHolder {
 public:
  LocalShardList() = default;

  void emplace_back(int64_t row_start, int64_t col_start, int64_t row_size, int64_t col_size, torch::Tensor tensor) {
    shards_.emplace_back(LocalShard{
        .row_start_ = row_start,
        .col_start_ = col_start,
        .row_size_ = row_size,
        .col_size_ = col_size,
        .tensor_ = tensor });
  }

  std::vector<LocalShard> shards_;
};

class PS : public torch::CustomClassHolder {
 public:
  PS(const std::string& table_name, c10::intrusive_ptr<LocalShardList> shards, int64_t col_size, const std::string& io_config) :
    table_name_(table_name), shards_(shards), col_size_(col_size), io_(io_config) {}

  void Fetch(torch::Tensor ids_to_fetch);
  void Evict(torch::Tensor ids_to_evict);

 private:
  std::string table_name_;
  c10::intrusive_ptr<LocalShardList> shards_;
  int64_t col_size_;
  details::IO io_;
};

} // namespace tde
