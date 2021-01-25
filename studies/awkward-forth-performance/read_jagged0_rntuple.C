R__LOAD_LIBRARY(ROOTNTuple)

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;

#define LENJAGGED0 1073741824
#define JAGGED0_CLUSTERSIZE 16777197

template <typename V, typename T>
void fillpages(T* array, V& view, int64_t& offset, int64_t length, int64_t shift) {
  int64_t current = 0;
  while (current < length) {
    T* data = (T*)view.fField.Map(offset + current);
    int32_t num = view.fField.fPrincipalColumn->fCurrentPage.GetNElements();
    int32_t skipped = (offset + current) - view.fField.fPrincipalColumn->fCurrentPage.GetGlobalRangeFirst();
    int32_t remaining = num - skipped;
    if (current + remaining > length) {
      remaining = length - current;
    }
    if (remaining > 0) {
      memcpy(&array[current + shift], data, remaining*sizeof(T));
    }
    current += remaining;
  }
  offset += current;
}

void read_jagged0_rntuple() {
  auto model = ROOT::Experimental::RNTupleModel::Create();

  std::string name = "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-rntuple-jagged0.root";

  auto begin_time = std::chrono::high_resolution_clock::now();

  auto ntuple = RNTupleReader::Open(std::move(model), "rntuple", name);
  auto view0 = ntuple->GetViewCollection("field");

  int64_t offset0 = 0;
  for (int64_t entry = 0;  entry < LENJAGGED0;  entry += JAGGED0_CLUSTERSIZE) {
    int64_t length = JAGGED0_CLUSTERSIZE;
    if (entry + length > LENJAGGED0) {
      length = LENJAGGED0 - entry;
    }
    float* rawcontent = new float[length];
    fillpages(rawcontent, view0, offset0, length, 0);

    delete [] rawcontent;
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  int64_t count_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - begin_time
  ).count();

  std::cout << "rntuple zlib0-jagged0.root " << (count_nanoseconds / 1e9) << " seconds" << std::endl;
}
