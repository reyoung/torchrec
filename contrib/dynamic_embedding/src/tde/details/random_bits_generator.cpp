#include "random_bits_generator.h"
#include "c10/macros/Macros.h"
#include "tde/details/bits_op.h"

namespace tde::details {

bool BitScanner::IsNextNBitsAllZero(uint16_t& n_bits) {
  if (n_bits == 0) {
    return true;
  }
  if (C10_UNLIKELY(array_idx_ == size_)) {
    return true;
  }

  auto val = array[array_idx_];
  val &= static_cast<uint64_t>(-1) >>
      bit_idx; // mask higher bits to zeros if already scan.
  uint16_t remaining_bits =
      static_cast<uint16_t>(sizeof(uint64_t) * 8) - bit_idx;
  
  if (val == 0) { // already all zero
    auto scanned_bits = std::min(remaining_bits, n_bits);
    n_bits -= scanned_bits;
    bit_idx += scanned_bits;
    if (bit_idx == sizeof(uint64_t) * 8) {
      bit_idx = 0;
      ++array_idx_;
    }
    if (n_bits == 0) {
      return true;
    } else {
      return IsNextNBitsAllZero(n_bits);
    }
  } else {
    uint16_t scanned_bits = Clz(val);
    if (scanned_bits >= n_bits + bit_idx) {
      bit_idx += n_bits;
      n_bits = 0;
      return true;
    }
    n_bits -= std::min(n_bits, remaining_bits);
    bit_idx = scanned_bits + 1;
    if (bit_idx > sizeof(uint64_t) * 8) {
      bit_idx = 0;
      ++array_idx_;
    }

    return false;
  }
}
BitScanner::BitScanner(size_t n) : array(new uint64_t[n]), size_(n) {}

constexpr static size_t k_n_random_elems = 8;

class NElemsRandom {
 public:
  explicit NElemsRandom(std::mt19937_64& engine) : engine_(engine) {}
  void operator()(tcb::span<uint64_t> elems) {
    for (auto& elem : elems) {
      elem = engine_();
    }
  }

 private:
  std::mt19937_64& engine_;
};

RandomBitsGenerator::RandomBitsGenerator()
    : engine_(std::random_device()()), scanner_(k_n_random_elems) {
  scanner_.ResetArray(NElemsRandom(engine_));
}
bool RandomBitsGenerator::IsNextNBitsAllZero(uint16_t n_bits) {
  bool ok = scanner_.IsNextNBitsAllZero(n_bits);
  if (n_bits != 0) { // scanner is end.
    scanner_.ResetArray(NElemsRandom(engine_));
  }
  if (!ok) {
    return false;
  }
  if (C10_UNLIKELY(n_bits != 0)) {
    return IsNextNBitsAllZero(n_bits);
  } else {
    return true;
  }
}

RandomBitsGenerator::~RandomBitsGenerator() = default;
} // namespace tde::details
