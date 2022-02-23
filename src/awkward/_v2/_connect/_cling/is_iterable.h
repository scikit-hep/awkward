#ifndef is_iterable_h
#define is_iterable_h

#include <type_traits>
#include <iterator>

namespace is_iterable_impl
{
    using std::begin;
    using std::end;

    template<class T>
    using check_specs = std::void_t<
        std::enable_if_t<std::is_same_v<
            decltype(begin(std::declval<T&>())), // has begin()
            decltype(end(std::declval<T&>()))    // has end()
        >>,                                      // ... begin() and end() are the same type ...
        decltype(*begin(std::declval<T&>()))     // ... which can be dereferenced
    >;

    template<class T, class = void>
    struct is_iterable
    : std::false_type
    {};

    template<class T>
    struct is_iterable<T, check_specs<T>>
    : std::true_type
    {};
}

template<class T>
using is_iterable = is_iterable_impl::is_iterable<T>;

template<class T>
constexpr bool is_iterable_v = is_iterable<T>::value;

#endif /* is_iterable_h */
