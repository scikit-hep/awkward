import awkward.forth
import awkward as ak
import numpy as np

form = ak.forms.Form.fromjson("""
{
    "class": "ListOffsetArray64",
    "offsets": "i64",
    "content": {
        "class": "RecordArray",
        "contents": {
            "x": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "node1"
            },
            "y": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node3"
                },
                "form_key": "node2"
            }
        }
    },
    "form_key": "node0"
}
""")

class TypedArrayBuilder:
    def __init__(self, form):
        # pretend we used 'form' to determine how to create 'vm' and 'fsm'

        self.form = form

        self.vm = awkward.forth.ForthMachine32("""
            input data
            output part0-node0-offsets int64
            output part0-node1-data float64
            output part0-node2-offsets int64
            output part0-node3-data int64

            0 part0-node0-offsets <- stack
            0 part0-node2-offsets <- stack

            0   ( initialize the total counter )
            0   ( initialize the node0 list counter )

            : fill-node1-goto-1
                0 data seek
                data d-> part0-node1-data
                0   ( initialize the node2 list counter )
                1   ( every routine ends with new FSM state )
            ;

            : fill-node0-inc-total-goto-0
                part0-node0-offsets +<- stack
                1+  ( increment the total counter )
                0   ( initialize the node0 list counter )
                0   ( every routine ends with new FSM state )
            ;

            : fill-node3-inc-node2-goto-1
                0 data seek
                data q-> part0-node3-data
                1+  ( increment the node2 list counter )
                1   ( every routine ends with new FSM state )
            ;

            : fill-node2-goto-2
                part0-node2-offsets +<- stack
                2   ( every routine ends with new FSM state )
            ;

            : inc-node0-goto-0
                1+  ( increment the node0 list counter )
                0   ( every routine ends with new FSM state )
            ;
        """)

        self.fsm = [
            # 0
            {
                "float64": "fill-node1-goto-1",
                "end_list": "fill-node0-inc-total-goto-0",
            },
            # 1
            {
                "int64": "fill-node3-inc-node2-goto-1",
                "end_list": "fill-node2-goto-2",
            },
            # 2
            {
                "end_record": "inc-node0-goto-0",
            },
        ]

        self.state = 0
        self.data = np.empty(8, np.uint8)
        self.vm.run({"data": self.data})

    def int64(self, x):
        self.data.view(np.int64)[0] = x
        word = self.fsm[self.state].get("int64")
        if word is None:
            raise ValueError("can't call 'int64' at this point in the process")
        self.vm.call(word)
        self.state = self.vm.stack_pop()

    def float64(self, x):
        self.data.view(np.float64)[0] = x
        word = self.fsm[self.state].get("float64")
        if word is None:
            raise ValueError("can't call 'float64' at this point in the process")
        self.vm.call(word)
        self.state = self.vm.stack_pop()

    def end_list(self):
        word = self.fsm[self.state].get("end_list")
        if word is None:
            raise ValueError("can't call 'end_list' at this point in the process")
        self.vm.call(word)
        self.state = self.vm.stack_pop()

    def end_record(self):
        word = self.fsm[self.state].get("end_record")
        if word is None:
            raise ValueError("can't call 'end_record' at this point in the process")
        self.vm.call(word)
        self.state = self.vm.stack_pop()

    def snapshot(self):
        return ak.from_buffers(self.form, self.vm.stack[0], self.vm.outputs)

    def debug_step(self):
        print("FSM state:", self.state)
        print("stack: ", builder.vm.stack)
        for k, v in builder.vm.outputs.items():
            print(k + ":", np.asarray(v))
        print("array:", self.snapshot())
        print()

# example = ak.Array([
#     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
#     [],
#     [{"x": 3.3, "y": [1, 2, 3]}],
# ])

builder = TypedArrayBuilder(form)
builder.debug_step()

builder.float64(1.1)
builder.debug_step()

builder.int64(1)
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.end_record()
builder.debug_step()

builder.float64(2.2)
builder.debug_step()

builder.int64(1)
builder.debug_step()

builder.int64(2)
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.end_record()
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.float64(3.3)
builder.debug_step()

builder.int64(1)
builder.debug_step()

builder.int64(2)
builder.debug_step()

builder.int64(3)
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.end_record()
builder.debug_step()

builder.end_list()
builder.debug_step()

assert builder.snapshot().tolist() == [
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
    [],
    [{"x": 3.3, "y": [1, 2, 3]}],
]
