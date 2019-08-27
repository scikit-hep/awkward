// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_UTIL_H_
#define AWKWARDCPU_UTIL_H_

#ifdef _MSC_VER
  #ifdef _WIN64
    typedef signed __int64 ssize_t;
  #else
    typedef signed   int   ssize_t;
  #endif
  typedef unsigned char  uint8_t;
  typedef signed   int   int32_t;
  typedef signed __int64 int64_t;
#else
  #include <cstdint>
#endif

extern "C" {
  typedef const char* Error;
  const Error kNoError = nullptr;

  typedef uint8_t byte;
  typedef int32_t IndexType;
  typedef unsigned char TagType;
  typedef int64_t ChunkOffsetType;
  typedef int64_t RefType;

  const IndexType       kMaxIndexType       =          2147483647;   // 2**31 - 1
  const ChunkOffsetType kMaxChunkOffsetType = 9223372036854775807;   // 2**63 - 1
}

#endif // AWKWARDCPU_UTIL_H_
