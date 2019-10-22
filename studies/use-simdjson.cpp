#include "../simdjson/singleheader/simdjson.h"
#include "../simdjson/singleheader/simdjson.cpp"

using namespace simdjson;

int main(int argc, char *argv[]) {
  padded_string unparsed = get_corpus("small-example.json");
  ParsedJson parsed = build_parsed_json(unparsed);
  return 0;
}
