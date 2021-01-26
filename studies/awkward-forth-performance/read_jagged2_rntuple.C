R__LOAD_LIBRARY(ROOTNTuple)

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;

#define LENJAGGED2 16777216
#define JAGGED2_CLUSTERSIZE 219310

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

void read_jagged2_rntuple(std::string which) {
  auto model = ROOT::Experimental::RNTupleModel::Create();

  std::string name = std::string("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/") + which + "-jagged2.root";

  auto ntuple = RNTupleReader::Open(std::move(model), "rntuple", name);
  auto view2 = ntuple->GetViewCollection("field");
  auto view1 = view2.GetViewCollection("std::vector<float>");
  auto view0 = view1.GetView<float>("float");

  auto begin_time = std::chrono::high_resolution_clock::now();

  int64_t offset2 = 0;
  int64_t offset1 = 0;
  int64_t offset0 = 0;
  for (int64_t entry = 0;  entry < LENJAGGED2;  entry += JAGGED2_CLUSTERSIZE) {
    int64_t length = JAGGED2_CLUSTERSIZE;
    if (entry + length > LENJAGGED2) {
      length = LENJAGGED2 - entry;
    }
    int32_t* rawoffsets2 = new int32_t[length + 1];
    rawoffsets2[0] = 0;
    fillpages(rawoffsets2, view2, offset2, length, 1);

    length = rawoffsets2[length];
    int32_t* rawoffsets1 = new int32_t[length + 1];
    rawoffsets1[0] = 0;
    fillpages(rawoffsets1, view1, offset1, length, 1);

    length = rawoffsets1[length];
    float* rawcontent = new float[length];
    fillpages(rawcontent, view0, offset0, length, 0);

    delete [] rawoffsets2;
    delete [] rawoffsets1;
    delete [] rawcontent;
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  int64_t count_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - begin_time
  ).count();

  std::cout << "rntuple " << which << "-jagged2 " << (count_nanoseconds / 1e9) << " seconds" << std::endl;
}
