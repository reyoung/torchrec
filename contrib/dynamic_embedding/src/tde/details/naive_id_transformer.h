#pragma once
#include <c10/util/flat_hash_map.h>
#include <memory>
#include <optional>
#include "tcb/span.hpp"

namespace tde::details {

namespace transform_default {

inline bool NoFilter(int64_t global_id) {
  return true;
}

template <typename Tag>
inline Tag NoUpdate(
    std::optional<Tag> tag,
    int64_t global_id,
    int64_t cache_id) {
  return tag.value_or(Tag{});
};

inline void NoFetch(int64_t global_id, int64_t cache_id) {}

} // namespace transform_default

template <typename T = uint32_t>
struct Bitmap {
  Bitmap(int64_t num_bits);
  int64_t NextFreeBit();
  void FreeBit(int64_t offset);

  static constexpr int64_t num_bits_per_mask = sizeof(T) * 8;

  const int64_t num_bits_;
  const int64_t num_masks_;
  std::unique_ptr<T[]> masks_;

  int64_t next_free_bit_;
};

template <typename Tag, typename T = uint32_t>
class NaiveIDTransformer {
 public:
  NaiveIDTransformer(int64_t num_embedding, int64_t embedding_offset);

  template <
      typename Filter = decltype(transform_default::NoFilter),
      typename Update = decltype(transform_default::NoUpdate<Tag>),
      typename Fetch = decltype(transform_default::NoFetch)>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Filter filter = transform_default::NoFilter,
      Update update = transform_default::NoUpdate<Tag>,
      Fetch fetch = transform_default::NoFetch);

  template <typename Callback>
  void ForEach(
      Callback callback = [](int64_t global_id, int64_t cache_id, Tag tag) {});

  void Evict(tcb::span<const int64_t> global_ids);

 private:
  const int64_t num_embedding_;
  const int64_t embedding_offset_;
  std::unique_ptr<Tag[]> tags_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
  Bitmap<T> bitmap_;
};

} // namespace tde::details

#include "tde/details/naive_id_transformer_impl.h"
