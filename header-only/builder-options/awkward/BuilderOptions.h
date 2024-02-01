// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_BUILDEROPTIONS_H_
#define AWKWARD_BUILDEROPTIONS_H_

#include <cassert>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>
#include <stdint.h>

namespace awkward {

  /// @class Options
  ///
  /// @brief Container for all configuration options needed by ArrayBuilder,
  /// GrowableBuffer, LayoutBuilder and the Builder subclasses.
  template <typename... OPTIONS>
  struct Options {
    static constexpr std::size_t value = sizeof...(OPTIONS);

    using OptionsPack = typename std::tuple<OPTIONS...>;

    template<std::size_t INDEX>
    using OptionType = std::tuple_element_t<INDEX, OptionsPack>;

    /// @brief Creates an Options tuple from a full set of parameters.
    Options(OPTIONS... options) : pars(options...) {}

    /// @brief The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    int64_t
    initial() const noexcept {
      return option<0>();
    }

    /// @brief The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    double
    resize() const noexcept {
      return option<1>();
    }

    /// @brief Access to all other options.
    template <std::size_t INDEX>
    const OptionType<INDEX>&
    option() const noexcept {
      return std::get<INDEX>(pars);
    }

    OptionsPack pars;
  };

  using BuilderOptions = Options<int64_t, double>;
}  // namespace awkward

#endif  // AWKWARD_BUILDEROPTIONS_H_
