import awkward1._ext
import awkward1._util

def copy_to(array, ptr_lib, highlevel=True, behavior=None):
    if(isinstance(array, awkward1.layout.Content) or
            isinstance(array, awkward1.layout.Index8) or
            isinstance(array, awkward1.layout.IndexU8) or
            isinstance(array, awkward1.layout.Index32) or
            isinstance(array, awkward1.layout.IndexU32) or
            isinstance(array, awkward1.layout.Index64)):
        return array.copy_to(ptr_lib)

    arr = awkward1.to_layout(array)
    if highlevel:
        return awkward1._util.wrap(arr.copy_to(ptr_lib), behavior)
    else:
        return arr.copy_to(ptr_lib)
