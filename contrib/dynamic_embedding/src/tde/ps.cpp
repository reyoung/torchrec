#include "tde/ps.h"
#include "tde/details/io.h"

namespace tde {

c10::intrusive_ptr<Notification> PS::Fetch(
    torch::Tensor ids_to_fetch,
    bool reinit,
    double weight_init_min,
    double weight_init_max) {
  TORCH_CHECK(ids_to_fetch.dim() == 2);
  std::vector<int64_t> col_ids{0};
  Filter(ids_to_fetch);
  c10::intrusive_ptr<Notification> notification = c10::make_intrusive<Notification>();
  if (cache_ids_to_fetch_or_evict_.empty()) {
    notification->Done();
    return notification;
  }
  uint32_t num_os_ids = os_ids_.size();
  io_.Pull(
      table_name_,
      global_ids_to_fetch_or_evict_,
      col_ids,
      num_os_ids,
      torch::kF32,
      [&](auto&& val) {
        TORCH_CHECK(val.size() == cache_ids_to_fetch_or_evict_.size());
        for (uint32_t i = 0; i < cache_ids_to_fetch_or_evict_.size(); ++i) {
          int64_t cache_id = cache_ids_to_fetch_or_evict_[i];
          auto& fetched = val[i];
          if (!fetched.defined()) {
            if (reinit) {
              std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
              tensors[0].uniform_(weight_init_min, weight_init_max);
              // optimizer states will be set to zero
              for (uint32_t j = 1; j < num_os_ids; ++j) {
                tensors[j].zero_();
              }
            }
            continue;
          }

          std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
          for (uint32_t j = 0; j < num_os_ids; ++j) {
            tensors[j].copy_(fetched.slice(0, j, j + 1));
          }
        }
        notification->Done();
      });
  return notification;
}

void PS::Filter(const torch::Tensor& tensor) {
  cache_ids_to_fetch_or_evict_.clear();
  global_ids_to_fetch_or_evict_.clear();
  TORCH_CHECK(tensor.is_contiguous());
  auto* ptr = tensor.data_ptr<int64_t>();
  int64_t numel = tensor.numel();
  for (int64_t i = 0; i < numel; i += 2, ptr += 2) {
    if (auto cache_id = ptr[1];
        std::any_of(shards_->begin(), shards_->end(), [&](auto&& shard) {
          return shard.Has(cache_id);
        })) {
      cache_ids_to_fetch_or_evict_.emplace_back(cache_id);
      global_ids_to_fetch_or_evict_.emplace_back(*ptr);
    }
  }
}

void PS::Evict(torch::Tensor ids_to_evict) {
  TORCH_CHECK(ids_to_evict.dim() == 2);
  std::vector<int64_t> col_ids{0};
  // remove this copy!
  Filter(ids_to_evict);
  if (global_ids_to_fetch_or_evict_.empty()) {
    return;
  }

  uint32_t num_os_ids = os_ids_.size();
  std::vector<uint64_t> offsets;
  offsets.reserve(
      global_ids_to_fetch_or_evict_.size() * num_os_ids * col_ids.size() + 1);
  offsets.emplace_back(0);
  std::vector<float> data(
      global_ids_to_fetch_or_evict_.size() * num_os_ids * col_ids.size() *
      col_size_);

  for (auto cache_id : cache_ids_to_fetch_or_evict_) {
    std::vector<torch::Tensor> tensors = GetTensorViews(cache_id);
    for (uint32_t j : os_ids_) {
      // this cause 2 copy. is this avoidable?
      torch::Tensor tensor = tensors[j].cpu();
      // need to change this when considering col
      memcpy(
          reinterpret_cast<uint8_t*>(data.data()) + offsets.back(),
          tensor.data_ptr<float>(),
          tensor.numel() * tensor.element_size());
      offsets.emplace_back(
          offsets.back() + tensor.numel() * tensor.element_size());
    }
  }

  details::Notification notification;
  io_.Push(
      table_name_,
      global_ids_to_fetch_or_evict_,
      col_ids,
      os_ids_,
      tcb::span{
          reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(float)},
      tcb::span{offsets.data(), offsets.size()},
      [&notification] { notification.Done(); });

  notification.Wait();
}

std::vector<torch::Tensor> PS::GetTensorViews(int64_t cache_id) {
  for (auto& shard : *shards_) {
    if (shard.Has(cache_id)) {
      return shard.GetTensorView(cache_id);
    }
  }
  TORCH_CHECK(false, "all local shards do not contain cache id ", cache_id);
}

} // namespace tde
