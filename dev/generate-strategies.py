from hypothesis import given, strategies as st
import os
import json

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def generate_strategies():
    with open(
        os.path.join(CURRENT_DIR,"..","hypothesis-tests-spec","strategies.py"),'w'
    ) as test_file:
        test_file.write("from hypothesis import given, strategies as st\n")
        test_file.write("from hypothesis.strategies import composite\n\n")
        with open(
            os.path.join(CURRENT_DIR,"..","constraints.json"),'r'
        ) as contraints_file:
            data=json.load(contraints_file)
            res="\treturn ("
            for kernels in data["kernels"]:
                test_file.write("@composite\n")
                r=0
                test_file.write("def "+"generate_"+kernels["name"]+"(draw,elements=st.integers()):\n")
                for args in kernels["arguments"]:
                    test_file.write("\t"+args["var"]+"="+"draw(st.")
                    if(len(args["constraints"])==0):
                        test_file.write(args["strategy_name"]+"())\n")
                    else:
                        num=0
                        test_file.write(args["strategy_name"]+"(")
                        if(args["strategy_name"]=="lists"):
                            test_file.write("elements")
                            num+=1
                        for cons in args["constraints"]:
                            if(num==0):
                                test_file.write(cons+"="+str(args["constraints"][cons]))
                                num+=1
                            else:
                                test_file.write(","+cons+"="+str(args["constraints"][cons]))
                                num+=1
                        test_file.write("))\n")
                    if(r==0):
                        res=res+args["var"]
                        r+=1
                    else:
                        res=res+","+args["var"]
                test_file.write(res+")")



generate_strategies()

