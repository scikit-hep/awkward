#include <stdio.h>
#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TInterpreter.h"

void make_jagged1_root() {
  int64_t events_per_basket = 64 * 1024 * 1024 / sizeof(float) / 8;

  gInterpreter->GenerateDictionary("vector<float>", "vector");

  auto f = new TFile("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root", "RECREATE");
  f->SetCompressionAlgorithm(1);
  f->SetCompressionLevel(9);

  auto t = new TTree("tree", "");

  std::vector<float> data1;
  t->Branch("branch", &data1, 300*1024*1024);
  t->SetAutoFlush(0);
  t->SetAutoSave(0);

  int64_t last1 = 999;
  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");
  FILE* offsets1 = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", "r");

  fread(&last1, sizeof(int64_t), 1, offsets1);

  float c = 3.14;
  int64_t o1 = 999;

  int64_t count = 0;
  while (fread(&o1, sizeof(int64_t), 1, offsets1) != 0) {
    data1.clear();
    for (int64_t k = 0;  k < (o1 - last1);  k++) {
      fread(&c, sizeof(float), 1, content);
      data1.push_back(c);
    }
    last1 = o1;

    t->Fill();
    count++;
    if (count % events_per_basket == 0) {
      t->Write();
    }
    // if (count % 10000 == 0) {
    //   std::cout << ((double)count / (3.0 * (double)events_per_basket)) << std::endl;
    // }
    // if (count == 3 * events_per_basket) {
    //   break;
    // }
  }

  t->Write();
  f->Close();
}
