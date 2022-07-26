#pragma once
#include <c10/util/flat_hash_map.h>
#include <memory>
#include <optional>
#include "tcb/span.hpp"

namespace tde::details {

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
  std::unique_ptr<Tag[]> tags_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
  Bitmap<T> bitmap_;
};

} // namespace tde::details

#include "tde/details/naive_id_transformer_impl.h"
