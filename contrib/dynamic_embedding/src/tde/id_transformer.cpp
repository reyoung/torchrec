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
  TORCH_CHECK(time >= 0);
  TORCH_CHECK(global_id_list->size() == cache_id_list->size());
  transformer_.strategy_.UpdateTime(static_cast<uint32_t>(time));
  int64_t total_num_embeddings = std::accumulate(
      global_id_list->begin(),
      global_id_list->end(),
      int64_t(0),
      [](int64_t v, auto&& tensor) -> int64_t { return v + tensor.numel(); });

  try {
    ids_to_fetch_.resize(2 * total_num_embeddings);
  } catch (std::bad_alloc& ex) {
    TORCH_CHECK(
        false,
        "bad allocate ",
        ex.what(),
        " the total_num_embeddings=",
        total_num_embeddings);
  }

  int64_t num_transformed = 0;
  std::atomic<int64_t> next_fetch_offset{0};
  for (int64_t i = 0; i < global_id_list->size(); ++i) {
    torch::Tensor global_ids = (*global_id_list)[i];
    torch::Tensor cache_ids = (*cache_id_list)[i];
    num_transformed += transformer_.Transform(
        tcb::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        tcb::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())},
        [&](int64_t global_id, int64_t cache_id) {
          int64_t offset = next_fetch_offset.fetch_add(1);
          ids_to_fetch_[2 * offset] = global_id;
          ids_to_fetch_[2 * offset + 1] = cache_id;
        });
  }
  bool success = num_transformed == total_num_embeddings;
  if (next_fetch_offset.load() == 0) {
    return c10::make_intrusive<TransformResult>(success, torch::Tensor{});
  }
  std::vector<int64_t> shape = {next_fetch_offset.load(), 2};

  torch::Tensor ids_to_fetch =
      torch::from_blob(
          ids_to_fetch_.data(),
          shape,
          torch::TensorOptions().dtype(c10::kLong).device(c10::kCPU))
          .clone()
          .data();
  TDE_DEBUG();
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
