#ifndef is_iterable_h
#define is_iterable_h

#include <type_traits>
#include <iterator>

namespace is_iterable_impl
{
    using std::begin;
    using std::end;

    template<typename... Ts> struct make_void { typedef void type;};
    template<typename... Ts> using void_t = typename make_void<Ts...>::type;

    template< bool B, class T = void >
    using enable_if_t = typename std::enable_if<B, T>::type;

    template< class T, class U >
    inline constexpr bool is_same_v = std::is_same<T, U>::value;

    template<class T>
    using check_specs = void_t<
        enable_if_t<is_same_v<
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
