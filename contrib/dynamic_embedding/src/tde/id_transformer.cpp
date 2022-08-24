#include "tde/id_transformer.h"
#include "tde/details/move_only_function.h"

namespace tde {

#define TDE_DEBUG() fprintf(stderr, __FILE__ ":%d\n", __LINE__)

IDTransformer::IDTransformer(int64_t num_embedding, nlohmann::json json)
    : json_(std::move(json)),
      transformer_(
          std::move(details::LXUStrategy(json_["lxu_strategy"])),
          num_embedding,
          json_["id_transformer"]),
      num_ids_to_fetch_(0) {}

c10::intrusive_ptr<TransformResult> IDTransformer::Transform(
    c10::intrusive_ptr<TensorList> global_id_list,
    c10::intrusive_ptr<TensorList> cache_id_list,
    int64_t time) {
  torch::NoGradGuard no_grad;
  TDE_DEBUG();
  TORCH_CHECK(time >= 0);
  TORCH_CHECK(global_id_list->size() == cache_id_list->size());
  TDE_DEBUG();
  transformer_.strategy_.UpdateTime(static_cast<uint32_t>(time));
  TDE_DEBUG();
  int64_t total_numel = 0;
  TDE_DEBUG();
  for (int64_t i = 0; i < global_id_list->size(); ++i) {
    total_numel += (*global_id_list)[i].numel();
  }
  TDE_DEBUG();
  try {
    ids_to_fetch_.resize(2 * total_numel);
  } catch (std::bad_alloc& ex) {
    TORCH_CHECK(
        false, "bad allocate ", ex.what(), " the total_numel=", total_numel);
  }
  TDE_DEBUG();
  int64_t num_transformed = 0;
  for (int64_t i = 0; i < global_id_list->size(); ++i) {
    torch::Tensor global_ids = (*global_id_list)[i];
    torch::Tensor cache_ids = (*cache_id_list)[i];
    TDE_DEBUG();
    num_transformed += transformer_.Transform(
        tcb::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        tcb::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())},
        [this](int64_t global_id, int64_t cache_id) {
          int64_t idx = num_ids_to_fetch_.fetch_add(1);
          ids_to_fetch_[2 * idx] = global_id;
          ids_to_fetch_[2 * idx + 1] = cache_id;
        });
  }
  TDE_DEBUG();
  bool success = num_transformed == total_numel;
  int64_t num_ids_to_fetch = num_ids_to_fetch_.load();
  if (num_ids_to_fetch == 0) {
    return c10::make_intrusive<TransformResult>(success, torch::Tensor{});
  }
  TDE_DEBUG();
  std::vector<int64_t> shape = {num_ids_to_fetch, 2};

  torch::Tensor ids_to_fetch = torch::from_blob(
                                   ids_to_fetch_.data(),
                                   shape,
                                   torch::dtype(torch::kLong))
                                   .clone()
                                   .data();
  TDE_DEBUG();
  num_ids_to_fetch_.store(0);
  return c10::make_intrusive<TransformResult>(success, ids_to_fetch);
}

torch::Tensor IDTransformer::Evict(int64_t num_to_evict) {
  torch::NoGradGuard no_grad;
  std::vector<int64_t> ids_to_evict = transformer_.Evict(num_to_evict);
  int64_t num_ids_to_evict = ids_to_evict.size() / 2;
  torch::Tensor evicted_ids_tensor =
      torch::tensor(ids_to_evict, torch::dtype(torch::kLong))
          .reshape({num_ids_to_evict, 2});
  return evicted_ids_tensor;
}

} // namespace tde
