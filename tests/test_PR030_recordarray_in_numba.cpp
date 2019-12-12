// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iostream>
#include <vector>

#include "awkward/Slice.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/FillableOptions.h"

namespace ak = awkward;

int main(int, char**) 
{

  // create fillable array
  ak::FillableArray myarray(ak::FillableOptions(1024, 2.0));

  // populate fillable array with lists
  myarray.beginlist();
  //myarray.boolean(true);
  myarray.integer(1);
  myarray.real(1.1);
  myarray.endlist();
  
  myarray.beginlist();
  //myarray.boolean(false);
  myarray.integer(2);
  myarray.real(2.2);
  myarray.endlist();

  myarray.beginlist();
  //myarray.boolean(true);
  myarray.integer(3);
  myarray.real(3.3);
  myarray.endlist();

  // take a snapshot 
  //std::shared_ptr<ak::Content> array = builder.snapshot();
 
  // saving to compare fillable array to std vector
  //std::vector<std::vector<std::vector<double>>> vector =
  //{{{true, 1, 1.1}, {false, 2, 2.2}, {true, 3, 3.3}}, {}, {}, {}};


  return 0;
}


