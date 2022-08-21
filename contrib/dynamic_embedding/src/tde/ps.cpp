#include "tde/ps.h"
#include "tde/details/io.h"
#include "tde/details/notification.h"

namespace tde {

void PS::Fetch(torch::Tensor ids_to_fetch) {
  std::vector<int64_t> col_ids{0};

  uint32_t num_offsets = ids_to_fetch.numel() + 1;
  std::vector<uint64_t> offsets(num_offsets);

  details::Notification notification;
  io_.Pull(
      table_name_,
      tcb::span{
          ids_to_fetch.template data_ptr<int64_t>(),
          static_cast<size_t>(ids_to_fetch.numel())},
      col_ids,
      1,
      torch::kF32,
      [&](std::vector<torch::Tensor> val) {
        TORCH_CHECK(val.size() == ids_to_fetch.numel());
        auto data_ptr = ids_to_fetch.template data_ptr<int64_t>();
        for (size_t i = 0; i < val.size(); ++i) {
          int64_t global_id = data_ptr[i];
          for (auto& shard : shards_->shards_) {
            torch::Tensor tensor = shard.GetTensorView(global_id);
            if (tensor.defined()) {
              tensor.copy_(val[i]);
              break;
            }
          }
        }
        notification.Done();
      });
  notification.Wait();
}

void PS::Evict(torch::Tensor ids_to_evict) {
  std::vector<int64_t> col_ids{0};
  std::vector<uint32_t> os_ids{0};

  uint32_t num_offsets = ids_to_evict.numel() + 1;
  std::vector<uint64_t> offsets(num_offsets);

  std::vector<float> data(ids_to_evict.numel() * col_ids.size() * os_ids.size() * col_size_);

  int64_t num_global_ids = ids_to_evict.numel();
  int64_t* global_ids_ptr = ids_to_evict.template data_ptr<int64_t>();
  for (int64_t i = 0; i < ids_to_evict.numel(); ++i) {
    int64_t global_id = global_ids_ptr[i];
    torch::Tensor tensor{};
    for (auto& shard : shards_->shards_) {
      tensor = shard.GetTensorView(global_id);
      if (tensor.defined()) {
        break;
      }
    }
    if (!tensor.defined()) {
      continue;
    }
    offsets[i + 1] = offsets[i] + tensor.numel() * tensor.element_size();
    // need to change this when considering col
    memcpy(reinterpret_cast<uint8_t*>(
        data.data()) + offsets[i],
        tensor.template data_ptr<float>(),
        tensor.numel() * tensor.element_size());
  }

  details::Notification notification;
  io_.Push(
      table_name_,
      tcb::span{
          ids_to_evict.template data_ptr<int64_t>(),
          static_cast<size_t>(ids_to_evict.numel())},
      col_ids,
      os_ids,
      tcb::span{
          reinterpret_cast<uint8_t*>(data.data()),
          data.size() * sizeof(float)},
      tcb::span{offsets.data(), offsets.size()},
      [&notification] { notification.Done(); });

  notification.Wait();
}

} // namespace tde
