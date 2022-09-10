// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class COMBINATIONS_ERRORS {
  FIXME  // message: "FIXME: awkward_combinations"
};

template <typename T>
__global__ void
awkward_combinations(T* toindex,
                     int64_t n,
                     bool replacement,
                     int64_t singlelen,
                     uint64_t invocation_index,
                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    RAISE_ERROR(COMBINATIONS_ERRORS::FIXME)
  }
}
