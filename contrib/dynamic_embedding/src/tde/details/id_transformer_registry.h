#pragma once
#include <variant>
#include "tde/details/mixed_lfu_lru_strategy.h"
#include "tde/details/multithreaded_id_transformer.h"
#include "tde/details/naive_id_transformer.h"
#include "tde/details/type_list.h"
namespace tde::details {

template <typename LXURecord>
using ComposableTransformers = type_list<NaiveIDTransformer<LXURecord>>;

template <typename UnderlyingTransformer>
using ComposeTransformers =
    type_list<MultiThreadedIDTransformer<UnderlyingTransformer>>;

using LXUStrategies = type_list<MixedLFULRUStrategy>;

namespace helpers {

template <typename L>
struct LXUStrategyRecordTypes;
template <typename Head, typename... Tail>
struct LXUStrategyRecordTypes<type_list<Head, Tail...>> {
  using type = cons_t<
      type_list<typename Head::lxu_record_t>,
      typename LXUStrategyRecordTypes<type_list<Tail...>>::type>;
};

template <>
struct LXUStrategyRecordTypes<type_list<>> {
  using type = type_list<>;
};

} // namespace helpers

using LXURecordTypes =
    typename helpers::LXUStrategyRecordTypes<LXUStrategies>::type;

namespace helpers {

template <typename T>
struct ComposableTransformersWithLXUStrategiesHelper;

template <typename Head, typename... Tail>
struct ComposableTransformersWithLXUStrategiesHelper<type_list<Head, Tail...>> {
  using type = cons_t<
      ComposableTransformers<typename Head::lxu_record_t>,
      typename ComposableTransformersWithLXUStrategiesHelper<
          type_list<Tail...>>::type>;
};

template <>
struct ComposableTransformersWithLXUStrategiesHelper<type_list<>> {
  using type = type_list<>;
};
} // namespace helpers

using ComposableTransformersWithLXUStrategies =
    typename helpers::ComposableTransformersWithLXUStrategiesHelper<
        LXUStrategies>::type;

namespace helpers {
template <typename T>
struct ComposeTransformersHelpers;

template <typename Head, typename... Tail>
struct ComposeTransformersHelpers<type_list<Head, Tail...>> {
  using type = cons_t<
      ComposeTransformers<Head>,
      typename ComposeTransformersHelpers<type_list<Tail...>>::type>;
};
template <>
struct ComposeTransformersHelpers<type_list<>> {
  using type = type_list<>;
};

} // namespace helpers

using ComposedTransformers = helpers::ComposeTransformersHelpers<
    ComposableTransformersWithLXUStrategies>::type;

using Transformers = typename cons<
    ComposableTransformersWithLXUStrategies,
    ComposedTransformers>::type;

} // namespace tde::details
