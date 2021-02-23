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
                "form_key": "node2"
            },
            "y": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node4"
                },
                "form_key": "node3"
            }
        },
        "form_key": "node1"
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
            output part0-node2-data float64
            output part0-node3-offsets int64
            output part0-node4-data int64

            : node4-int64
                ( no pause because this is first after a begin_list )

                {int64} = if
                    0 data seek
                    data q-> part0-node4-data
                else
                    halt
                then
            ;

            : node3-list
                pause
                {begin_list} <> if
                    halt
                then

                0   ( initialize the node3 list counter )
                begin
                    pause
                    dup {end_list} = if
                        drop
                        part0-node3-offsets +<- stack ( absorbs node3 counter )
                        exit
                    else
                        node4-int64
                        1+   ( increment the node3 list counter )
                    then
                again
            ;

            : node2-float64
                pause
                {float64} = if
                    0 data seek
                    data d-> part0-node2-data
                else
                    halt
                then
            ;

            : node1-record
                ( no pause because this is first after a begin_list )

                {begin_record} <> if
                    halt
                then

                node2-float64
                node3-list

                pause
                {end_record} <> if
                    halt
                then
            ;

            : node0-list
                pause
                {begin_list} <> if
                    halt
                then

                0   ( initialize the node0 list counter )
                begin
                    pause
                    dup {end_list} = if
                        drop
                        part0-node0-offsets +<- stack ( absorbs node0 counter )
                        exit
                    else
                        node1-record
                        1+   ( increment the node0 list counter )
                    then
                again
            ;

            0 part0-node0-offsets <- stack
            0 part0-node3-offsets <- stack

            0    ( initialize the total counter )
            begin
                node0-list
                1+   ( increment the total counter )
            again
        """.format(int64=0, float64=1, begin_list=2, end_list=3, begin_record=4, end_record=5))

        self.data = np.empty(8, np.uint8)
        self.vm.run({"data": self.data})

    def int64(self, x):
        self.data.view(np.int64)[0] = x
        self.vm.stack_push(0)
        self.vm.resume()

    def float64(self, x):
        self.data.view(np.float64)[0] = x
        self.vm.stack_push(1)
        self.vm.resume()

    def begin_list(self):
        self.vm.stack_push(2)
        self.vm.resume()

    def end_list(self):
        self.vm.stack_push(3)
        self.vm.resume()

    def begin_record(self):
        self.vm.stack_push(4)
        self.vm.resume()

    def end_record(self):
        self.vm.stack_push(5)
        self.vm.resume()

    def snapshot(self):
        return ak.from_buffers(self.form, self.vm.stack[0], self.vm.outputs)

    def debug_step(self):
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

builder.begin_list()
builder.debug_step()

builder.begin_record()
builder.debug_step()

builder.float64(1.1)
builder.debug_step()

builder.begin_list()
builder.debug_step()

builder.int64(1)
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.end_record()
builder.debug_step()

builder.begin_record()
builder.debug_step()

builder.float64(2.2)
builder.debug_step()

builder.begin_list()
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

builder.begin_list()
builder.debug_step()

builder.end_list()
builder.debug_step()

builder.begin_list()
builder.debug_step()

builder.begin_record()
builder.debug_step()

builder.float64(3.3)
builder.debug_step()

builder.begin_list()
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
