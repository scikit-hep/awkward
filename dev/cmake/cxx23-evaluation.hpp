// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// MSVC reports its selected mode through _MSVC_LANG without requiring /Zc:__cplusplus.
// C++23 support is partial on some compilers; this checks the language mode,
// not conformance to every C++23 feature. CMake may use /std:c++latest on MSVC.
#if defined(_MSVC_LANG)
static_assert(_MSVC_LANG > 202002L,
              "C++23 evaluation requires a post-C++20 language mode");
#else
static_assert(__cplusplus > 202002L,
              "C++23 evaluation requires a post-C++20 language mode");
#endif
