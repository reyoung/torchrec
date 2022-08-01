#pragma once
#include <optional>
#include "lexy/action/parse.hpp"
#include "lexy/callback.hpp"
#include "lexy/dsl.hpp"
#include "lexy/input/string_input.hpp"
#include "lexy_ext/report_error.hpp"

/**
 * A simple URL/extendable parser
 */
namespace tde::details::url_parser {

struct Auth {
  std::string username_;
  std::optional<std::string> password_;
};

struct DefaultParamRule {
  static constexpr auto rule = lexy::dsl::capture(lexy::dsl::any);
  static constexpr auto value = lexy::as_string<std::string>;
};

template <typename Param = DefaultParamRule>
struct Url {
  std::optional<Auth> auth_;
  std::string host_;
  std::optional<uint16_t> port_;
  std::optional<typename decltype(Param::value)::return_type> param_;
};

namespace rules {
namespace dsl = lexy::dsl;
using AuthValue = Auth;
template <typename P>
using UrlValue = Url<P>;
struct UrlToken {
  static constexpr auto rule = dsl::percent_sign >>
          dsl::integer<uint8_t>(dsl::n_digits<2, dsl::hex>) |
      dsl::capture(dsl::ascii::alpha_digit_underscore);

  static constexpr auto value = lexy::callback<char>([](auto&& v) -> char {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, uint8_t>) {
      return v;
    } else {
      return v[0];
    }
  });
};

struct UrlString {
  static constexpr auto rule = dsl::list(dsl::p<UrlToken>);
  static constexpr auto value = lexy::as_string<std::string>;
};

struct Auth {
  static constexpr auto rule = dsl::p<UrlString> >>
      dsl::opt(dsl::lit_c<':'> >> dsl::p<UrlString>) >> dsl::lit_c<'@'>;
  static constexpr auto value = lexy::construct<AuthValue>;
};
struct Host {
  static constexpr auto rule =
      dsl::list(dsl::p<UrlToken> | dsl::capture(dsl::lit_c<'.'>));
  static constexpr auto value = lexy::as_string<std::string>;
};

template <typename Param = DefaultParamRule>
struct Url {
  using ctor_type = UrlValue<Param>;

  static constexpr auto rule =
      dsl::opt(dsl::lookahead(dsl::lit_c<'@'>, dsl::newline) >> dsl::p<Auth>) +
      dsl::p<Host> +
      dsl::opt(dsl::lit_c<':'> >> dsl::integer<uint16_t>(dsl::digits<>)) +
      dsl::opt(dsl::lit_c<'/'> >> dsl::p<Param>);

  static constexpr auto value = lexy::construct<ctor_type>;
};

} // namespace rules

template <typename ParamRule>
inline std::optional<Url<ParamRule>> ParseUrl(std::string_view url_str) {
  // NOTE: cannot using rule = xxx here. It seems a bug of clang or lexy.
  struct _rule : public rules::Url<ParamRule> {};

  auto input = lexy::string_input(url_str);
  auto parsed = lexy::parse<_rule>(input, lexy_ext::report_error);
  if (parsed.has_value()) {
    return parsed.value();
  } else {
    return std::nullopt;
  }
}

} // namespace tde::details::url_parser
