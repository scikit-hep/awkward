# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# Evaluation-only hook, enabled with CMAKE_PROJECT_INCLUDE after compiler detection.
# Force every C++ translation unit to verify the requested standard actually took
# effect, without interfering with CMake's compiler-identification probes.
if(MSVC)
  add_compile_options("/FI${CMAKE_CURRENT_LIST_DIR}/cxx23-evaluation.hpp")
else()
  add_compile_options(-include "${CMAKE_CURRENT_LIST_DIR}/cxx23-evaluation.hpp")
endif()
