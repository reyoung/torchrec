#pragma once
#include <algorithm>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

template <typename LXURecord, typename BitMap, typename Hash>
inline CachelineIDTransformer<LXURecord, BitMap, Hash>::CachelineIDTransformer(
    int64_t num_embedding)
    : num_groups_(num_embedding * 2 / kGroupSize + 1),
      cache_values_(new CacheValue[num_groups_ * kGroupSize]),
      bitmap_(num_embedding) {
  std::fill(
      cache_values_.get(),
      cache_values_.get() + num_groups_ * kGroupSize,
      CacheValue{.global_id_ = 0, .tagged_cache_id_ = 0});
}

template <typename LXURecord, typename BitMap, typename Hash>
template <
    typename Filter,
    typename CacheIDTransformer,
    typename Update,
    typename Fetch>
inline int64_t CachelineIDTransformer<LXURecord, BitMap, Hash>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Filter filter,
    CacheIDTransformer cache_id_transformer,
    Update update,
    Fetch fetch) {
  int64_t num_transformed = 0;
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    if (!filter(global_id)) {
      continue;
    }
    auto [group_id, intra_id] = FindGroupIndex(global_id);

    bool need_eviction = true;
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    for (int64_t k = 0; k < kGroupSize; k++) {
      int64_t offset = group_id * kGroupSize + (intra_id + k) % kGroupSize;
      auto& cache_value = cache_values_[offset];
      if (cache_value.is_filled()) {
        if (cache_value.global_id_ == global_id) {
          cache_id = cache_value.cache_id();
          cache_value.lxu_record_ =
              update(cache_value.lxu_record_, global_id, cache_id);
          need_eviction = false;
          break;
        }
      } else {
        // The transformer is full.
        if (C10_UNLIKELY(bitmap_.Full())) {
          break;
        }
        auto stored_cache_id = bitmap_.NextFreeBit();
        cache_id = cache_id_transformer(stored_cache_id);
        cache_value.global_id_ = global_id;
        cache_value.set_cache_id(cache_id);
        cache_value.lxu_record_ =
            update(cache_value.lxu_record_, global_id, cache_id);
        fetch(global_id, cache_id);
        need_eviction = false;
        break;
      }
    }
    if (need_eviction)
      break;
    cache_ids[i] = cache_id;
    num_transformed++;
  }
  return num_transformed;
}

template <typename LXURecord, typename BitMap, typename Hash>
template <typename Callback>
inline void CachelineIDTransformer<LXURecord, BitMap, Hash>::ForEach(
    Callback callback) {
  for (int64_t i = 0; i < num_groups_; ++i) {
    for (int64_t j = 0; j < kGroupSize; ++j) {
      int64_t offset = i * kGroupSize + j;
      auto& cache_value = cache_values_[offset];
      if (cache_value.is_filled()) {
        callback(
            cache_value.global_id_,
            cache_value.cache_id(),
            cache_value.lxu_record_);
      }
    }
  }
}

template <typename LXURecord, typename BitMap, typename Hash>
inline void CachelineIDTransformer<LXURecord, BitMap, Hash>::Evict(
    tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto [group_id, intra_id] = FindGroupIndex(global_id);

    for (int64_t k = 0; k < kGroupSize; k++) {
      int64_t offset = group_id * kGroupSize + (intra_id + k) % kGroupSize;
      auto& cache_value = cache_values_[offset];
      if (!cache_value.is_filled()) {
        break;
      } else {
        if (cache_value.global_id_ == global_id) {
          bitmap_.FreeBit(cache_value.cache_id());
          cache_value.set_empty();
          break;
        }
      }
    }
  }
}

template <typename LXURecord, typename BitMap, typename Hash>
inline auto CachelineIDTransformer<LXURecord, BitMap, Hash>::Iterator() const
    -> MoveOnlyFunction<std::optional<record_t>()> {
  int64_t i = 0;
  return [i, this]() mutable -> std::optional<record_t> {
    for (; i < num_groups_ * kGroupSize; ++i) {
      auto& cache_value = cache_values_[i];
      if (cache_value.is_filled()) {
        auto record = record_t{
            .global_id_ = cache_value.global_id_,
            .cache_id_ = cache_value.cache_id(),
            .lxu_record_ = cache_value.lxu_record_,
        };
        ++i;
        return record;
      }
    }
    return {};
  };
}

} // namespace tde::details
