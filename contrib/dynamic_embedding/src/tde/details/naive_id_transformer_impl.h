#pragma once
#include <string.h>

namespace tde::details {

template <typename T>
Bitmap<T>::Bitmap(int64_t num_bits)
    : num_bits_(num_bits),
      num_masks_((num_bits + num_bits_per_mask - 1) / num_bits_per_mask),
      masks_(new T[num_masks_]),
      next_free_bit_(0) {
  std::fill(masks_.get(), masks_.get() + num_masks_, -1);
}

template <typename T>
int64_t Bitmap<T>::NextFreeBit() {
  int64_t next_free_bit = next_free_bit_;
  int64_t mask_offset = next_free_bit / num_bits_per_mask;
  T mask = masks_[mask_offset];
  masks_[mask_offset] = mask & (mask - 1);
  while (masks_[mask_offset] == 0 && mask_offset < num_masks_) {
    mask_offset++;
  }
  mask = masks_[mask_offset];
  if (mask) {
    if constexpr (num_bits_per_mask <= 32) {
      next_free_bit_ = mask_offset * num_bits_per_mask + __builtin_ctz(mask);
    } else {
      next_free_bit_ = mask_offset * num_bits_per_mask + __builtin_ctzll(mask);
    }
  } else {
    next_free_bit_ = num_bits_;
  }

  return next_free_bit;
}

template <typename T>
void Bitmap<T>::FreeBit(int64_t offset) {
  int64_t mask_offset = offset / num_bits_per_mask;
  int64_t bit_offset = offset % num_bits_per_mask;
  masks_[mask_offset] |= 1 << bit_offset;
  if (offset < next_free_bit_) {
    next_free_bit_ = offset;
  }
}

template <typename Tag, typename T>
NaiveIDTransformer<Tag, T>::NaiveIDTransformer(
    int64_t num_embedding,
    int64_t embedding_offset)
    : num_embedding_(num_embedding),
      embedding_offset_(embedding_offset),
      tags_(new Tag[num_embedding]),
      bitmap_(num_embedding) {}

template <typename Tag, typename T>
template <typename Filter, typename Update, typename Fetch>
int64_t NaiveIDTransformer<Tag, T>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Filter filter,
    Update update,
    Fetch fetch) {
  int64_t num_transformed = 0;
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    if (!filter(global_id)) {
      continue;
    }
    auto iter = global_id2cache_id_.find(global_id);
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    if (iter != global_id2cache_id_.end()) {
      cache_id = iter->second;
    } else {
      // The transformer is full.
      if (bitmap_.next_free_bit_ >= bitmap_.num_bits_) {
        break;
      }
      cache_id = bitmap_.NextFreeBit();
      global_id2cache_id_.emplace(global_id, cache_id);
      fetch(global_id, cache_id + embedding_offset_);
    }
    cache_ids[i] = cache_id + embedding_offset_;
    tags_[cache_id] =
        update(tags_[cache_id], global_id, cache_id + embedding_offset_);
    num_transformed++;
  }
  return num_transformed;
}

template <typename Tag, typename T>
template <typename Callback>
void NaiveIDTransformer<Tag, T>::ForEach(Callback callback) {
  for (auto [global_id, cache_id] : global_id2cache_id_) {
    Tag tag = tags_[cache_id];
    callback(global_id, cache_id, tag);
  }
}

template <typename Tag, typename T>
void NaiveIDTransformer<Tag, T>::Evict(tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto iter = global_id2cache_id_.find(global_id);
    if (iter == global_id2cache_id_.end()) {
      continue;
    }
    int64_t cache_id = iter->second;
    global_id2cache_id_.erase(iter);
    bitmap_.FreeBit(cache_id);
  }
}

} // namespace tde::details
