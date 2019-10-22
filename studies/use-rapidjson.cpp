// c++ --std=c++11 use-rapidjson.cpp -o use-rapidjson && ./use-rapidjson

#include <cstdio>
#include <iostream>

#include "../rapidjson/include/rapidjson/reader.h"
#include "../rapidjson/include/rapidjson/filereadstream.h"
 
namespace rj = rapidjson;

class MyHandler: public rj::BaseReaderHandler<rj::UTF8<>, MyHandler> {
  public:
    bool Null() { std::cout << "Null()" << std::endl; return true; }
    bool Bool(bool b) { std::cout << "Bool(" << std::boolalpha << b << ")" << std::endl; return true; }
    bool Int(int i) { std::cout << "Int(" << i << ")" << std::endl; return true; }
    bool Uint(unsigned u) { std::cout << "Uint(" << u << ")" << std::endl; return true; }
    bool Int64(int64_t i) { std::cout << "Int64(" << i << ")" << std::endl; return true; }
    bool Uint64(uint64_t u) { std::cout << "Uint64(" << u << ")" << std::endl; return true; }
    bool Double(double d) { std::cout << "Double(" << d << ")" << std::endl; return true; }
    bool String(const char* str, rj::SizeType length, bool copy) { 
        std::cout << "String(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
        return true;
    }
    bool StartObject() { std::cout << "StartObject()" << std::endl; return true; }
    bool Key(const char* str, rj::SizeType length, bool copy) { 
        std::cout << "Key(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
        return true;
    }
    bool EndObject(rj::SizeType memberCount) { std::cout << "EndObject(" << memberCount << ")" << std::endl; return true; }
    bool StartArray() { std::cout << "StartArray()" << std::endl; return true; }
    bool EndArray(rj::SizeType elementCount) { std::cout << "EndArray(" << elementCount << ")" << std::endl; return true; }
};
 
int main() {
    const char json[] = " { \"hel\\u2012lo\" : \"wo\\u2012rld\", \"t\" : true , \"f\" : false, \"n\": null, \"i\":123, \"pi\": 3.1416, \"a\":[1, 2, 3, 4] } ";
  
    MyHandler handler;
    rj::Reader reader;

    rj::StringStream ss(json);
    // FILE* fp = fopen("small-example.json", "rb");
    // char readBuffer[65536];
    // rj::FileReadStream ss(fp, readBuffer, sizeof(readBuffer));

    reader.Parse(ss, handler);

    return 0;
}
