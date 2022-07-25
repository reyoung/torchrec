#pragma once

namespace tde::details {

template <typename Tag>
NaiveIDTransformer<Tag>::NaiveIDTransformer(
    const int64_t num_embedding,
    const int64_t embedding_offset)
    : num_embedding_(num_embedding),
      embedding_offset_(embedding_offset),
      next_cache_id_(0),
      tags_(new Tag[num_embedding]) {}

template <typename Tag>
template <typename Filter, typename Update, typename Fetch>
int64_t NaiveIDTransformer<Tag>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Filter filter,
    Update update,
    Fetch fetch) {
  int64_t num_transformed = 0;
  for (size_t i = 0; i < global_ids.size(); ++i) {
    // The transformer is full.
    if (next_cache_id_ == num_embedding_)
      break;

    const int64_t global_id = global_ids[i];
    if (filter(global_id)) {
      auto iter = global_id2cache_id_.find(global_id);
      // cache_id is in [0, num_embedding)
      int64_t cache_id;
      if (iter != global_id2cache_id_.end()) {
        cache_id = iter->second;
      } else {
        cache_id = next_cache_id_++;
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

template <typename Tag>
template <typename Callback>
void NaiveIDTransformer<Tag>::ForEach(Callback callback) {
  for (auto& pair : global_id2cache_id_) {
    int64_t global_id = pair.first;
    int64_t cache_id = pair.second;
    Tag tag = tags_[cache_id];
    callback(global_id, cache_id, tag);
  }
}

template <typename Tag>
void NaiveIDTransformer<Tag>::Evict(tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    global_id2cache_id_.erase(global_id);
  }
}

} // namespace tde::details
