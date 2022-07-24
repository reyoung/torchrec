#pragma once
#include <atomic>
#include <optional>
#include "tde/details/move_only_function.h"
#include "tde/details/random_bits_generator.h"
namespace tde::details {

class MixedLFULRUStrategy {
 public:
  using ExtendedValueType = uint32_t;

  /**
   * @param min_used_freq_power min usage is 2^min_used_freq_power. Set this to
   * avoid recent values evict too fast.
   */
  explicit MixedLFULRUStrategy(uint16_t min_used_freq_power = 5);

  void UpdateTime(uint32_t time);

  ExtendedValueType Transform(std::optional<ExtendedValueType> val);

  /**
   * Analysis all ids and returns the num_elems that are most need to evict.
   * @param id_visitor Returns each global_id to ExtValue pair. Returns nullopt
   * when at ends.
   * @param num_elems_to_evict
   * @return
   */
  static std::vector<int64_t> Evict(
      MoveOnlyFunction<std::optional<std::pair<int64_t, ExtendedValueType>>()>
          id_visitor,
      uint64_t num_elems_to_evict);

  // Record should only be used in unittest or internally.
  struct Record {
    uint16_t freq_power_ : 5;
    uint32_t time_ : 27;
  };

 private:
  static_assert(sizeof(Record) == sizeof(ExtendedValueType));

  RandomBitsGenerator generator_;
  std::atomic<uint32_t> time_{};
  uint16_t min_lfu_power_;
};

} // namespace tde::details
