// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename A>
__global__ void
awkward_UnionArray_validity(const int8_t* tags,
                            const A* index,
                            int64_t length,
                            int64_t numcontents,
                            const int64_t* lencontents,
                            uint64_t invocation_index,
                            uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      auto tag = tags[thread_id];
      auto idx = index[thread_id];
      if ((tag < 0)) {
        err->str = "tags[i] < 0";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
      if ((idx < 0)) {
        err->str = "index[i] < 0";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
      if ((tag >= numcontents)) {
        err->str = "tags[i] >= len(contents)";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
      auto lencontent = lencontents[tag];
      if ((idx >= lencontent)) {
        err->str = "index[i] >= len(content[tags[i]])";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
    }
  }
}
