# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.layout
import awkward1.operations.convert

class Array(object):
    def __init__(self, data, type=None, copy=False):
        if isinstance(data, awkward1.layout.Content):
            layout = data
        elif isinstance(data, Array):
            layout = data.layout
        elif isinstance(data, numpy.ndarray):
            layout = awkward1.operations.convert.fromnumpy(data).layout
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data).layout
        else:
            layout = awkward1.operations.convert.fromiter(data).layout
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("could not convert data into an awkward1.Array")
        self.layout = layout

    # def __repr__(self):
    #     halfway = len(self.layout) // 2
    #     def forward(array):
    #         HERE

    def __len__(self):
        return len(self.layout)

    def __getitem__(self, where):
        layout = self.layout[where]
        if isinstance(layout, awkward1.layout.Content):
            return awkward1.Array(layout)
        elif isinstance(layout, awkward1.layout.Record):
            return awkward1.Record(layout)
        else:
            return layout

class Record(object):
    pass
