import os
from hypothesis import given,settings,strategies as st
import yaml


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@given(st.lists(st.integers().filter(lambda x: x > 1),min_size=4,max_size=4, unique=True))
@settings(max_examples=1)
def gentest(x):
    input=[{'input':x}]
    with open(
        os.path.join(CURRENT_DIR, "..","test_fuzz.yml"), "w"
    ) as fuzz:
        doc=yaml.dump(input,fuzz)


if __name__ == "__main__":
    gentest()