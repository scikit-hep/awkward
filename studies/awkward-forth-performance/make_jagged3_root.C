#include <stdio.h>
#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TInterpreter.h"

void make_jagged3_root() {
  int64_t events_per_basket = 64 * 1024 * 1024 / sizeof(float) / 8 / 8 / 8;

  gInterpreter->GenerateDictionary("vector<vector<vector<float> > >", "vector");

  auto f = new TFile("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged3.root", "RECREATE");
  f->SetCompressionAlgorithm(1);
  f->SetCompressionLevel(0);

  auto t = new TTree("tree", "");

  std::vector<std::vector<std::vector<float>>> data3;
  t->Branch("branch", &data3, 300*1024*1024);
  t->SetAutoFlush(0);
  t->SetAutoSave(0);

  int64_t last1 = 999;
  int64_t last2 = 999;
  int64_t last3 = 999;
  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");
  FILE* offsets1 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", "r");
  FILE* offsets2 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets2.int64", "r");
  FILE* offsets3 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets3.int64", "r");

  fread(&last1, sizeof(int64_t), 1, offsets1);
  fread(&last2, sizeof(int64_t), 1, offsets2);
  fread(&last3, sizeof(int64_t), 1, offsets3);

  float c = 3.14;
  int64_t o1 = 999;
  int64_t o2 = 999;
  int64_t o3 = 999;

  int64_t count = 0;
  while (fread(&o3, sizeof(int64_t), 1, offsets3) != 0) {
    data3.clear();
    for (int64_t i = 0;  i < (o3 - last3);  i++) {
      std::vector<std::vector<float>> data2;
      fread(&o2, sizeof(int64_t), 1, offsets2);
      for (int64_t j = 0;  j < (o2 - last2);  j++) {
        std::vector<float> data1;
        fread(&o1, sizeof(int64_t), 1, offsets1);
        for (int64_t k = 0;  k < (o1 - last1);  k++) {
          fread(&c, sizeof(float), 1, content);
          data1.push_back(c);
        }
        data2.push_back(data1);
        last1 = o1;
      }
      last2 = o2;
      data3.push_back(data2);
    }

    // std::cout << "[";
    // for (auto it3 : data3) {
    //   std::cout << "[";
    //   for (auto it2 : it3) {
    //     std::cout << "[";
    //     for (auto it1 : it2) {
    //       std::cout << it1 << " ";
    //     }
    //     std::cout << "]";
    //   }
    //   std::cout << "]";
    // }
    // std::cout << "]" << std::endl;

    last3 = o3;

    t->Fill();
    count++;
    if (count % events_per_basket == 0) {
      t->Write();
    }
  }

  t->Write();
  f->Close();
}
