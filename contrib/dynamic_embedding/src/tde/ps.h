#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/io.h"
#include "tde/tensor_list.h"

namespace tde {

struct LocalShard {
  int64_t row_start_;
  int64_t col_start_;
  int64_t row_size_;
  int64_t col_size_;
  c10::intrusive_ptr<TensorList> tensors_;

  std::vector<torch::Tensor> GetTensorView(int64_t global_id) {
    if (global_id < row_start_ || global_id >= row_start_ + row_size_) {
      return {};
    }
    std::vector<torch::Tensor> result;
    result.reserve(tensors_->size());
    for (int64_t i = 0; i < tensors_->size(); ++i) {
      result.emplace_back((*tensors_)[i].slice(
          0, global_id - row_start_, global_id - row_start_ + 1));
    }
    return result;
  }
};

class LocalShardList : public torch::CustomClassHolder {
 public:
  LocalShardList() = default;

  void emplace_back(
      int64_t row_start,
      int64_t col_start,
      int64_t row_size,
      int64_t col_size,
      c10::intrusive_ptr<TensorList> tensors) {
    shards_.emplace_back(LocalShard{
        .row_start_ = row_start,
        .col_start_ = col_start,
        .row_size_ = row_size,
        .col_size_ = col_size,
        .tensors_ = tensors});
  }

  std::vector<LocalShard> shards_;
};

class PS : public torch::CustomClassHolder {
 public:
  PS(const std::string& table_name,
     c10::intrusive_ptr<LocalShardList> shards,
     int64_t col_size,
     int64_t num_optimizer_stats,
     const std::string& io_config)
      : table_name_(table_name),
        shards_(shards),
        col_size_(col_size),
        os_ids_(num_optimizer_stats),
        io_(io_config) {
    for (int64_t i = 0; i < num_optimizer_stats; ++i) {
      os_ids_[i] = i;
    }
  }

  void Fetch(torch::Tensor ids_to_fetch, bool reinit, double weight_init_min, double weight_init_max);
  void Evict(torch::Tensor ids_to_evict);

 private:
  std::vector<torch::Tensor> GetTensorViews(int64_t global_id);

  std::string table_name_;
  c10::intrusive_ptr<LocalShardList> shards_;
  int64_t col_size_;
  std::vector<uint32_t> os_ids_;
  details::IO io_;
};

} // namespace tde
