R__LOAD_LIBRARY(ROOTNTuple)

#include <stdio.h>
#include <iostream>
#include <vector>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;

#define JAGGED3_CLUSTERSIZE 28508

void make_jagged3_rntuple() {
  auto model = RNTupleModel::Create();
  std::shared_ptr<std::vector<std::vector<std::vector<float>>>> field = model->MakeField<std::vector<std::vector<std::vector<float>>>>("field");

  RNTupleWriteOptions options;
  options.SetCompression(0);

  std::string name = "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-rntuple-jagged3.root";
  auto rntuple = RNTupleWriter::Recreate(std::move(model), "rntuple", name, options);
  rntuple->fClusterSizeEntries = JAGGED3_CLUSTERSIZE;

  int64_t last1 = 999;
  int64_t last2 = 999;
  int64_t last3 = 999;
  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");
  FILE* offsets1 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", "r");
  FILE* offsets2 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets2.int64", "r");
  FILE* offsets3 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets3.int64", "r");

  ssize_t
  tmp = fread(&last1, sizeof(int64_t), 1, offsets1);
  tmp = fread(&last2, sizeof(int64_t), 1, offsets2);
  tmp = fread(&last3, sizeof(int64_t), 1, offsets3);

  float c = 3.14;
  int64_t o1 = 999;
  int64_t o2 = 999;
  int64_t o3 = 999;

  while (fread(&o3, sizeof(int64_t), 1, offsets3) != 0) {
    field.get()->clear();
    for (int64_t i = 0;  i < (o3 - last3);  i++) {
      std::vector<std::vector<float>> data2;
      tmp = fread(&o2, sizeof(int64_t), 1, offsets2);
      for (int64_t j = 0;  j < (o2 - last2);  j++) {
        std::vector<float> data1;
        tmp = fread(&o1, sizeof(int64_t), 1, offsets1);
        for (int64_t k = 0;  k < (o1 - last1);  k++) {
          tmp = fread(&c, sizeof(float), 1, content);
          data1.push_back(c);
        }
        data2.push_back(data1);
        last1 = o1;
      }
      last2 = o2;
      field.get()->push_back(data2);
    }
    last3 = o3;

    rntuple->Fill();
  }
}
