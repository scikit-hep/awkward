import kernels
import os
import json

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_awkward_IndexedArray_validity():
    f= open(os.path.join(CURRENT_DIR,"..","test-data.json"),)
    data=json.load(f)
    for d in data["datas"]:
        x=d["input"]
        y=d["output"]
        result=kernels.awkward_IndexedArray_validity(*tuple(x))
        assert result==y

test_awkward_IndexedArray_validity()