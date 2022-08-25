#pragma once
#include "torch/torch.h"
namespace tde::details {

template <typename UnderlyingTransformer>
template <typename UnderlyingTransformerCreator>
MultiThreadedIDTransformer<UnderlyingTransformer>::MultiThreadedIDTransformer(
    int64_t num_embedding,
    size_t num_threads,
    UnderlyingTransformerCreator creator)
    : num_threads_(num_threads), thread_pool_(new ThreadPool(num_threads)) {
  transformers_.reserve(num_threads);
  embedding_offsets_.reserve(num_threads);
  int64_t embedding_per_transformer = num_embedding / num_threads;
  int64_t embedding_offset = 0;
  for (size_t i = 0; i < num_threads; i++) {
    embedding_offsets_.emplace_back(embedding_offset);
    transformers_.emplace_back(creator(
        i == num_threads - 1 ? num_embedding - embedding_offset
                             : embedding_per_transformer));
    embedding_offset += embedding_per_transformer;
  }
}

template <typename UnderlyingTransformer>
template <typename Update, typename Fetch>
bool MultiThreadedIDTransformer<UnderlyingTransformer>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Update update,
    Fetch fetch) {
  std::vector<std::future<bool>> futures;
  futures.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    futures.emplace_back(std::move(thread_pool_->Enqueue([&, this, i] {
      return transformers_[i].Transform(
          global_ids,
          cache_ids,
          [n = num_threads_, i](int64_t global_ids) {
            return global_ids % static_cast<int64_t>(n) == i;
          },
          [i, this](int64_t cache_id) {
            return cache_id + embedding_offsets_[i];
          },
          update,
          fetch);
    })));
  }

  return std::accumulate(
      futures.begin(), futures.end(), true, [](bool ok, auto&& fut) -> bool {
        // make ok right side to avoid shortcut
        return fut.get() && ok;
      })
}

template <typename UnderlyingTransformer>
void MultiThreadedIDTransformer<UnderlyingTransformer>::Evict(
    tcb::span<const int64_t> global_ids) {
  for (size_t i = 0; i < num_threads_; ++i) {
    transformers_[i].Evict(global_ids);
  }
}

template <typename UnderlyingTransformer>
auto MultiThreadedIDTransformer<UnderlyingTransformer>::Iterator()
    -> MoveOnlyFunction<std::optional<record_t>()> {
  auto iter = transformers_.begin();
  auto iterator = iter->Iterator();
  return [iter,
          this,
          iterator = std::move(iterator)]() mutable -> std::optional<record_t> {
    auto opt = iterator();
    while (!opt.has_value()) {
      ++iter;
      if (iter == transformers_.end()) {
        return {};
      }
      iterator = iter->Iterator();
      opt = iterator();
    }
    return opt;
  };
}

} // namespace tde::details
