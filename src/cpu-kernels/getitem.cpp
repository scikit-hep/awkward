// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/getitem.h"

template <typename T>
void awkward_regularize_rangeslice(T& start, T& stop, bool posstep, bool hasstart, bool hasstop, T length) {
  if (posstep) {
    if (!hasstart)          start = 0;
    else if (start < 0)     start += length;
    if (start < 0)          start = 0;
    if (start > length)     start = length;

    if (!hasstop)           stop = length;
    else if (stop < 0)      stop += length;
    if (stop < 0)           stop = 0;
    if (stop > length)      stop = length;
    if (stop < start)       stop = start;
  }

  else {
    if (!hasstart)          start = length - 1;
    else if (start < 0)     start += length;
    if (start < -1)         start = -1;
    if (start > length - 1) start = length - 1;

    if (!hasstop)           stop = -1;
    else if (stop < 0)      stop += length;
    if (stop < -1)          stop = -1;
    if (stop > length - 1)  stop = length - 1;
    if (stop > start)       stop = start;
  }
}

void awkward_regularize_rangeslice_64(int64_t& start, int64_t& stop, bool posstep, bool hasstart, bool hasstop, int64_t length) {
  awkward_regularize_rangeslice<int64_t>(start, stop, posstep, hasstart, hasstop, length);
}

Error awkward_getitem() {
  return "not implemented";
}
