# Experimental C++23 migration

The `ianna/experimental-cxx23` branch requires C++23 through the
`awkward-parent` and header-only CMake interface targets. This applies to the
CPU kernels, libawkward, Python bindings, and consumers of the header-only
CMake targets. Configure with CMake 3.20 or newer and a compiler that supports its
`cxx_std_23` feature. A lower requested language standard cannot downgrade
these targets.

The initial compatibility review covers all 147 CPU kernel translation units.
They do not use removed `std::iterator`, `std::auto_ptr`, or
`std::random_shuffle` interfaces. This is a language compatibility review, not
a claim that all existing arithmetic and bounds behavior has been audited.

The migration preserves the generated C kernel ABI, kernel specification,
algorithms, and package versions. The existing scalar loops, type traits,
comparators, and standard algorithms remain valid C++23; switching algorithms
to ranges or parallel execution is separate work requiring correctness and
performance evidence. No third-party RapidJSON sources are modified.

The C++23 evaluation workflow builds GCC, Clang, AppleClang, and MSVC, checks
the selected language mode, runs the generated reference and compiled kernel
suites plus Forth regressions, and builds and runs the header-only tests.
This checks compiler compatibility, not complete implementation of every
C++23 library feature. CUDA/NVRTC and external ROOT interpreters have separate
language settings; this branch does not require their runtime compilers to
accept C++23.

Before proposing this as the upstream minimum, check the compiler matrix,
release wheel platforms (including Emscripten), downstream C++ consumers, and
performance. Local AppleClang validation alone does not establish support for
all those environments.

This migration was prepared with AI assistance.

Local validation (AppleClang, macOS arm64, Python 3.12):

- Native wheel built with C++23 selected for all 170 compiled translation units.
- Reference kernel suites: 9,008 passed, 336 skipped.
- Compiled kernel suites and Forth regressions: 13420 passed, 336 skipped in 14.82s.
- Debug header-only tests: all six passed.
- Workflow actionlint and whitespace checks passed.
