#include <stdio.h>
#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

void make_jagged0_root() {
  int64_t events_per_basket = 64 * 1024 * 1024 / sizeof(float);

  auto f = new TFile("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root", "RECREATE");
  f->SetCompressionAlgorithm(1);
  f->SetCompressionLevel(9);

  auto t = new TTree("tree", "");

  float data0;
  t->Branch("branch", &data0, 300*1024*1024);
  t->SetAutoFlush(0);
  t->SetAutoSave(0);

  FILE* content = fopen("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", "r");

  int64_t count = 0;
  while (fread(&data0, sizeof(float), 1, content) != 0) {
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
