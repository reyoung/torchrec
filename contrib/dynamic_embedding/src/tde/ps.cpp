#include "tde/ps.h"
#include "tde/details/io.h"
#include "tde/details/notification.h"

namespace tde {

void PS::Fetch(torch::Tensor ids_to_fetch, bool reinit, double weight_init_min, double weight_init_max) {
  TORCH_CHECK(ids_to_fetch.dim() == 2);
  std::vector<int64_t> col_ids{0};
  // remove this copy!
  torch::Tensor global_ids = ids_to_fetch.slice(1, 0, 1).contiguous();
  torch::Tensor cache_ids = ids_to_fetch.slice(1, 1, 2).contiguous();
  uint32_t num_global_ids = global_ids.numel();
  uint32_t num_os_ids = os_ids_.size();

  details::Notification notification;
  io_.Pull(
      table_name_,
      tcb::span{global_ids.template data_ptr<int64_t>(), num_global_ids},
      col_ids,
      num_os_ids,
      torch::kF32,
      [&](std::vector<torch::Tensor> val) {
        TORCH_CHECK(val.size() == num_global_ids);
        auto global_data_ptr = global_ids.template data_ptr<int64_t>();
        auto cache_data_ptr = cache_ids.template data_ptr<int64_t>();
        for (uint32_t i = 0; i < num_global_ids; ++i) {
          if (!val[i].defined()) {
            if (reinit) {
              int64_t cache_id = cache_data_ptr[i];
              std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
              tensors[0].uniform_(weight_init_min, weight_init_max);
              // optimizer states will be set to zero
              for (uint32_t j = 1; j < num_os_ids; ++j) {
                tensors[j].zero_();
              }
            }
            continue;
          }
          int64_t global_id = global_data_ptr[i];
          int64_t cache_id = cache_data_ptr[i];
          std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
          if (tensors.size() == 0) {
            continue;
          }
          for (uint32_t j = 0; j < num_os_ids; ++j) {
            tensors[j].copy_(val[i][j]);
          }
        }
        notification.Done();
      });
  notification.Wait();
}

void PS::Evict(torch::Tensor ids_to_evict) {
  TORCH_CHECK(ids_to_evict.dim() == 2);
  std::vector<int64_t> col_ids{0};
  // remove this copy!
  torch::Tensor global_ids = ids_to_evict.slice(1, 0, 1).contiguous();
  torch::Tensor cache_ids = ids_to_evict.slice(1, 1, 2).contiguous();

  uint32_t num_global_ids = global_ids.numel();
  uint32_t num_os_ids = os_ids_.size();
  uint32_t num_offsets = num_global_ids * num_os_ids * col_ids.size() + 1;
  std::vector<uint64_t> offsets(num_offsets);

  std::vector<float> data(
      num_global_ids * num_os_ids * col_ids.size() * col_size_);

  int64_t* global_ids_ptr = global_ids.template data_ptr<int64_t>();
  int64_t* cache_ids_ptr = cache_ids.template data_ptr<int64_t>();
  for (uint32_t i = 0; i < num_global_ids; ++i) {
    int64_t global_id = global_ids_ptr[i];
    int64_t cache_id = cache_ids_ptr[i];
    std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
    if (tensors.size() == 0) {
      continue;
    }
    for (uint32_t j : os_ids_) {
      // this cause 2 copy. is this avoidable?
      torch::Tensor tensor = tensors[j].cpu();
      uint32_t id = i * num_os_ids + j;
      offsets[id + 1] = offsets[id] + tensor.numel() * tensor.element_size();
      // need to change this when considering col
      memcpy(
          reinterpret_cast<uint8_t*>(data.data()) + offsets[id],
          tensor.template data_ptr<float>(),
          tensor.numel() * tensor.element_size());
    }
  }

  details::Notification notification;
  io_.Push(
      table_name_,
      tcb::span{global_ids.template data_ptr<int64_t>(), num_global_ids},
      col_ids,
      os_ids_,
      tcb::span{
          reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(float)},
      tcb::span{offsets.data(), offsets.size()},
      [&notification] { notification.Done(); });

  notification.Wait();
}

std::vector<torch::Tensor> PS::GetTensorViews(int64_t global_id) {
  std::vector<torch::Tensor> tensors;
  for (auto& shard : shards_->shards_) {
    tensors = shard.GetTensorView(global_id);
    if (tensors.size() != 0) {
      break;
    }
  }
  return tensors;
}

} // namespace tde
