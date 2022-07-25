#pragma once
#include <c10/util/flat_hash_map.h>
#include <memory>
#include <optional>
#include "tcb/span.hpp"

namespace tde::details {

template <typename BitMask>
struct BitMaskGroup {
  BitMaskGroup(int64_t total_bits);
  int64_t NextFreeBit();
  void FreeBit(int64_t offset);

  static constexpr int64_t B = sizeof(BitMask) * 8;

  const int64_t total_bits_;
  const int64_t total_bitmasks_;
  std::unique_ptr<BitMask[]> bitmasks_;

  int64_t next_free_bit_;
  bool full_;
};

template <typename Tag, typename BitMask>
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
  std::unique_ptr<Tag[]> tags_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
  BitMaskGroup<BitMask> bitmasks_;
};

} // namespace tde::details

#include "tde/details/naive_id_transformer_impl.h"
