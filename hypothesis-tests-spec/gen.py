import kernels
import os
import json
import awkward as ak
from hypothesis import strategies as st, given, settings
from hypothesis.strategies import composite

data=[]

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@composite
def strat(draw):
    lencontent=draw(st.integers(min_value=1,max_value=10))
    index=draw(st.lists(st.integers().filter(lambda x:x > -1 and x<lencontent),min_size=1,max_size=5))
    length=len(index)
    isoption=False
    return (index,length,lencontent,isoption)

@given(strat())
@settings(max_examples=10)
def find(ex):
    x=kernels.awkward_IndexedArray_validity(*ex)
    io={"input":"","output":""}
    io["input"]=ex
    io["output"]=x
    data.append(io)

def test():
    with open(os.path.join(CURRENT_DIR,"..","test-data.json"),"w") as js:
        ex={"datas":data}
        json.dump(ex,js,indent=1)    

#find = given(run())(settings(max_examples=5)(find))

if __name__ == "__main__":
    find()
    test()