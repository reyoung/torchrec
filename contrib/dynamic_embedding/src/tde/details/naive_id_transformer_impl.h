#pragma once

namespace tde::details {

template <typename BitMask>
BitMaskGroup<BitMask>::BitMaskGroup(int64_t total_bits)
    : total_bits_(total_bits),
      total_bitmasks_((total_bits + B - 1) / B),
      bitmasks_(new BitMask[total_bitmasks_]),
      next_free_bit_(0),
      full_(false) {
  memset(
      bitmasks_.get(),
      static_cast<BitMask>(-1),
      sizeof(BitMask) * total_bitmasks_);
}

template <typename BitMask>
int64_t BitMaskGroup<BitMask>::NextFreeBit() {
  int64_t next_free_bit = next_free_bit_;
  int64_t bitmask_offset = next_free_bit / B;
  BitMask bitmask = bitmasks_[bitmask_offset];
  bitmasks_[bitmask_offset] = bitmask & (bitmask - 1);
  while (bitmasks_[bitmask_offset] == 0 && bitmask_offset < total_bitmasks_) {
    bitmask_offset++;
  }
  bitmask = bitmasks_[bitmask_offset];
  if (bitmask) {
    next_free_bit_ = bitmask_offset * B + __builtin_ctz(bitmask);
    if (next_free_bit_ >= total_bits_) {
      full_ = true;
    }
  } else {
    full_ = true;
  }

  return next_free_bit;
}

template <typename BitMask>
void BitMaskGroup<BitMask>::FreeBit(int64_t offset) {
  int64_t bitmask_offset = offset / B;
  int64_t bit_offset = offset % B;
  bitmasks_[bitmask_offset] |= 1 << bit_offset;
}

template <typename Tag, typename BitMask>
NaiveIDTransformer<Tag, BitMask>::NaiveIDTransformer(
    const int64_t num_embedding,
    const int64_t embedding_offset)
    : num_embedding_(num_embedding),
      embedding_offset_(embedding_offset),
      tags_(new Tag[num_embedding]),
      bitmasks_(num_embedding) {}

template <typename Tag, typename BitMask>
template <typename Filter, typename Update, typename Fetch>
int64_t NaiveIDTransformer<Tag, BitMask>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Filter filter,
    Update update,
    Fetch fetch) {
  int64_t num_transformed = 0;
  for (size_t i = 0; i < global_ids.size(); ++i) {
    const int64_t global_id = global_ids[i];
    if (filter(global_id)) {
      auto iter = global_id2cache_id_.find(global_id);
      // cache_id is in [0, num_embedding)
      int64_t cache_id;
      if (iter != global_id2cache_id_.end()) {
        cache_id = iter->second;
      } else {
        // The transformer is full.
        if (bitmasks_.full_)
          break;
        cache_id = bitmasks_.NextFreeBit();
        global_id2cache_id_.emplace(global_id, cache_id);
        fetch(global_id, cache_id + embedding_offset_);
      }
      cache_ids[i] = cache_id + embedding_offset_;
      tags_[cache_id] =
          update(tags_[cache_id], global_id, cache_id + embedding_offset_);
      num_transformed++;
    }
  }
  return num_transformed;
}

template <typename Tag, typename BitMask>
template <typename Callback>
void NaiveIDTransformer<Tag, BitMask>::ForEach(Callback callback) {
  for (auto& pair : global_id2cache_id_) {
    int64_t global_id = pair.first;
    int64_t cache_id = pair.second;
    Tag tag = tags_[cache_id];
    callback(global_id, cache_id, tag);
  }
}

template <typename Tag, typename BitMask>
void NaiveIDTransformer<Tag, BitMask>::Evict(
    tcb::span<const int64_t> global_ids) {
  int64_t min_evicted_cache_id = num_embedding_;
  for (const int64_t global_id : global_ids) {
    auto iter = global_id2cache_id_.find(global_id);
    if (iter != global_id2cache_id_.end()) {
      int64_t cache_id = iter->second;
      if (cache_id < min_evicted_cache_id) {
        min_evicted_cache_id = iter->second;
      }
      global_id2cache_id_.erase(iter);
      bitmasks_.FreeBit(cache_id);
    }
  }
  bitmasks_.next_free_bit_ = min_evicted_cache_id;
  bitmasks_.full_ = false;
}

} // namespace tde::details
