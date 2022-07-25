#pragma once
#include <c10/util/flat_hash_map.h>
#include <memory>
#include <optional>
#include "tcb/span.hpp"

namespace tde::details {

template <typename Tag>
class NaiveIDTransformer {
 public:
  NaiveIDTransformer(
      const int64_t num_embedding,
      const int64_t embedding_offset);

  template <typename Filter, typename Update, typename Fetch>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Filter filter = [](int64_t global_id) -> bool { return true; },
      Update update = [](Tag tag, int64_t global_id, int64_t cache_id) -> Tag {
        return tag;
      },
      Fetch fetch = [](int64_t global_id, int64_t cache_id) {});

  template <typename Callback>
  void ForEach(
      Callback callback = [](int64_t global_id, int64_t cache_id, Tag tag) {});

  void Evict(tcb::span<const int64_t> global_ids);

 private:
  const int64_t num_embedding_;
  const int64_t embedding_offset_;
  int64_t next_cache_id_;
  std::unique_ptr<Tag[]> tags_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
};

} // namespace tde::details

#include "tde/details/naive_id_transformer_impl.h"
