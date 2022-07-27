#pragma once
#include "torch/torch.h"
namespace tde::details {

template <typename T>
MultiThreadedIDTransformer<T>::MultiThreadedIDTransformer(
    int64_t num_embedding,
    size_t num_threads)
    : num_threads_(num_threads), thread_pool_(num_threads) {
  transformers_.reserve(num_threads);
  embedding_offsets_.reserve(num_threads);
  int64_t embedding_per_transformer = num_embedding / num_threads;
  int64_t embedding_offset = 0;
  for (size_t i = 0; i < num_threads; i++) {
    embedding_offsets_.emplace_back(embedding_offset);
    transformers_.emplace_back(
        i == num_threads - 1 ? num_embedding - embedding_offset
                             : embedding_per_transformer);
    embedding_offset += embedding_per_transformer;
  }
}

template <typename T>
template <typename Update, typename Fetch>
int64_t MultiThreadedIDTransformer<T>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Update update,
    Fetch fetch) {
  std::vector<std::future<int64_t>> futures;
  futures.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    futures.emplace_back(std::move(thread_pool_.Enqueue([&, this, i] {
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
  int64_t num_transformed = 0;
  for (size_t i = 0; i < num_threads_; ++i) {
    num_transformed += futures[i].get();
  }

  return num_transformed;
}

template <typename T>
void MultiThreadedIDTransformer<T>::Evict(tcb::span<const int64_t> global_ids) {
  for (size_t i = 0; i < num_threads_; ++i) {
    transformers_[i].Evict(global_ids);
  }
}

template <typename T>
MoveOnlyFunction<std::optional<std::pair<int64_t, typename T::lxu_record_t>>()>
MultiThreadedIDTransformer<T>::CreateIDVisitor() {
  auto iter = transformers_.begin();
  MoveOnlyFunction<std::optional<std::pair<int64_t, LXURecord>>()> id_visitor =
      iter->CreateIDVisitor();
  return [iter, this, id_visitor = std::move(id_visitor)]() mutable {
    auto opt = id_visitor();
    while (!opt.has_value()) {
      iter++;
      if (iter != transformers_.end()) {
        id_visitor = std::move(iter->CreateIDVisitor());
        opt = id_visitor();
      } else {
        return opt;
      }
    }
    return opt;
  };
}

} // namespace tde::details
