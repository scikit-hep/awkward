#include <string>
#include <map>


struct ArrayBuffers {
    std::map<std::string, void*> buffers;
    std::map<std::string, size_t> buffer_nbytes;
    std::string form;
    size_t length;
};

template<typename T>
ArrayBuffers snapshot_builder(const T &builder);

ArrayBuffers create_demo_array();
