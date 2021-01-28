R__LOAD_LIBRARY(ROOTNTuple)

#include <stdio.h>
#include <iostream>
#include <vector>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;

#define JAGGED0_CLUSTERSIZE 16777197

void make_jagged0_rntuple(std::string compress_str, int compress_int) {
  auto model = RNTupleModel::Create();
  std::shared_ptr<float> field = model->MakeField<float>("field");

  RNTupleWriteOptions options;
  options.SetCompression(compress_int);

  std::string name = std::string("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/") + compress_str + "-rntuple-jagged0.root";
  auto rntuple = RNTupleWriter::Recreate(std::move(model), "rntuple", name, options);
  rntuple->fClusterSizeEntries = JAGGED0_CLUSTERSIZE;

  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");

  float c = 3.14;

  while (fread(&c, sizeof(float), 1, content) != 0) {
    *field = c;

    rntuple->Fill();
  }
}
