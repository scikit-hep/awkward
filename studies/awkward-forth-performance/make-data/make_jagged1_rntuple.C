R__LOAD_LIBRARY(ROOTNTuple)

#include <stdio.h>
#include <iostream>
#include <vector>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;

#define JAGGED1_CLUSTERSIZE 1342176

void make_jagged1_rntuple(std::string compress_str, int compress_int) {
  auto model = RNTupleModel::Create();
  std::shared_ptr<std::vector<float>> field = model->MakeField<std::vector<float>>("field");

  RNTupleWriteOptions options;
  options.SetCompression(compress_int);

  std::string name = std::string("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/") + compress_str + "-rntuple-jagged1.root";
  auto rntuple = RNTupleWriter::Recreate(std::move(model), "rntuple", name, options);
  rntuple->fClusterSizeEntries = JAGGED1_CLUSTERSIZE;

  int64_t last1 = 999;
  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");
  FILE* offsets1 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", "r");

  ssize_t
  tmp = fread(&last1, sizeof(int64_t), 1, offsets1);

  float c = 3.14;
  int64_t o1 = 999;

  while (fread(&o1, sizeof(int64_t), 1, offsets1) != 0) {
    field.get()->clear();
    for (int64_t k = 0;  k < (o1 - last1);  k++) {
      tmp = fread(&c, sizeof(float), 1, content);
      field.get()->push_back(c);
    }
    last1 = o1;

    rntuple->Fill();
  }
}
