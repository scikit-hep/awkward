from collections.abc import MutableMapping
import collections

counter = [0]
def generate():
    counter[0] += 1
    dict_generate = {1: 'Ferrari', 2: 'Mercedes', 3: 'Red Bull', 4: 'McLaren'}
    return dict_generate[counter[0]]

class Cache():
    def __init__(self, )

class VirtualArray():
    def __init__(self, generator, cache=None, cache_key=None):
        assert callable(generator)
        if cache is not None:
            assert isinstance(cache, collections.abc.MutableMapping)
        if cache_key is not None:
            assert isinstance(cache_key, str)
        self.generator = generator
        self.cache = cache
        self.cache_key = cache_key

    @staticmethod
    def random(minlen, choices):
        raise NotImplementedError("FIXME!")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, where):
        return self.cache[where]

    def __repr__(self):
        return '{}, D({})'.format(super(VirtualArray, self).__repr__(), 
                                  self.__dict__)

    def xml(self, indent="", pre="", post=""):
        raise NotImplementedError("FIXME!")

    def array(self):
        if(self.cache != None):
            array = self.cache[self.cache_key]
        else:
            array = self.generator()
        if(self.cache != None):
            self.cache_key = str(counter[0])
        return array

va = VirtualArray(generate, {"ak0": 'Ferrari', "ak1": 'Mercedes', "ak2": 'Red Bull', "ak3": 'McLaren'}, "ak2")
print(va.array())
