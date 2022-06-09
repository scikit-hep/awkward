# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# TODO: count => form_next_id or form_last_id
# TODO: dec => declarations
# TODO: keys => form_keys
# TODO: con => container
# TODO: more informative error messages

import json

import numpy as np

import awkward._v2 as ak
import awkward.forth


class _ReachedEndofArrayError(Exception):
    pass


class ReadAvroFT:
    def __init__(self, file, show_code=False):
        self.data = file
        self.blocks = 0
        self.marker = 0
        self.is_primitive = False
        numbytes = 1024
        self.temp_header = bytearray()
        self.metadata = {}

        while True:
            try:
                self.temp_header += self.data.read(numbytes)
                if not self.check_valid():
                    raise ak._v2._util.error(
                        TypeError("invalid Avro file: first 4 bytes are not 'Obj1'")
                    )
                pos = 4
                pos, self.pairs = self.decode_varint(4, self.temp_header)
                self.pairs = self.decode_zigzag(self.pairs)
                if self.pairs < 0:
                    pos, self.header_size = self.decode_varint(pos, self.temp_header)
                    self.header_size = self.decode_zigzag(self.pairs)
                    self.pairs = abs(self.pairs)
                pos = self.cont_spec(pos)
                break

            except _ReachedEndofArrayError:  # noqa: AK101
                numbytes *= 2

        ind = 2
        self.update_pos(pos)
        exec_code = []
        init_code = [": init-out\n"]
        header_code = "input stream \n"

        (
            self.form,
            self.exec_code,
            self.count,
            dec,
            keys,
            init_code,
            con,
        ) = self.rec_exp_json_code(
            self.metadata["avro.schema"], exec_code, ind, 0, [], [], init_code, {}
        )

        first_iter = True
        init_code.append(";\n")
        self.update_pos(17)
        header_code = header_code + "".join(dec)
        init_code = "".join(init_code)
        exec_code.insert(0, "0 do \n")
        exec_code.append("\nloop")
        exec_code = "".join(exec_code)

        forth_code = f"""
                {header_code}
                {init_code}
                {exec_code}"""
        if show_code:
            print(forth_code)  # noqa: T201

        machine = awkward.forth.ForthMachine64(forth_code)
        while True:
            try:
                pos, num_items, len_block = self.decode_block()
                temp_data = self.data.read(len_block)
                if len(temp_data) < len_block:
                    raise _ReachedEndofArrayError  # noqa: AK101
                self.update_pos(len_block)
            except _ReachedEndofArrayError:  # noqa: AK101
                break

            if first_iter:
                machine.begin({"stream": np.frombuffer(temp_data, dtype=np.uint8)})
                machine.stack_push(num_items)
                machine.call("init-out")
                machine.resume()
                first_iter = False
            else:
                machine.begin_again(
                    {"stream": np.frombuffer(temp_data, dtype=np.uint8)}, True
                )
                machine.stack_push(num_items)
                machine.resume()

            self.update_pos(16)

        for elem in keys:
            if "offsets" in elem:
                con[elem] = machine.output_Index64(elem)
            else:
                con[elem] = machine.output_NumpyArray(elem)

        self.outarr = ak.from_buffers(self.form, self.blocks, con)

    def update_pos(self, pos):
        self.marker += pos
        self.data.seek(self.marker)

    def decode_block(self):
        temp_data = self.data.read(10)
        pos, info = self.decode_varint(0, temp_data)
        info1 = self.decode_zigzag(info)
        self.update_pos(pos)
        self.blocks += int(info1)
        temp_data = self.data.read(10)
        pos, info = self.decode_varint(0, temp_data)
        info2 = self.decode_zigzag(info)
        self.update_pos(pos)

        return pos, info1, info2

    def cont_spec(self, pos):
        temp_count = 0
        while temp_count < self.pairs:
            pos, dat = self.decode_varint(pos, self.temp_header)
            dat = self.decode_zigzag(dat)
            key = self.temp_header[pos : pos + int(dat)]
            pos = pos + int(dat)
            if len(key) < int(dat):
                raise _ReachedEndofArrayError  # noqa: AK101

            pos, dat = self.decode_varint(pos, self.temp_header)
            dat = self.decode_zigzag(dat)
            val = self.temp_header[pos : pos + int(dat)]
            pos = pos + int(dat)
            if len(val) < int(dat):
                raise _ReachedEndofArrayError  # noqa: AK101
            if key == b"avro.schema":
                self.metadata[key.decode()] = json.loads(val.decode())
            else:
                self.metadata[key.decode()] = val

            temp_count += 1

        return pos

    def check_valid(self):
        init = self.temp_header[0:4]
        if len(init) < 4:
            raise _ReachedEndofArrayError  # noqa: AK101
        return init == b"Obj\x01"

    def decode_varint(self, pos, _data):
        shift = 0
        result = 0
        while True:
            if pos >= len(_data):
                raise _ReachedEndofArrayError  # noqa: AK101
            i = _data[pos]
            pos += 1
            result |= (i & 0x7F) << shift
            shift += 7
            if not i & 0x80:
                break

        return pos, result

    def decode_zigzag(self, n):
        return (n >> 1) ^ (-(n & 1))

    def dum_dat(self, dtype, count):
        if dtype["type"] == "int":
            return f"0 node{count}-data <- stack "
        elif dtype["type"] == "long":
            return f"0 node{count}-data <- stack "
        elif dtype["type"] == "float":
            return f"0 node{count}-data <- stack "
        elif dtype["type"] == "double":
            return f"0 node{count}-data <- stack "
        elif dtype["type"] == "boolean":
            return f"0 node{count}-data <- stack "
        elif dtype["type"] == "bytes":
            return f"1 node{count}-offsets +<- stack 97 node{count+1}-data <- stack "
        elif dtype["type"] == "string":
            return f"0 node{count}-offsets +<- stack "
        elif dtype["type"] == "enum":
            return f"0 node{count}-index <- stack "
        else:
            raise AssertionError  # noqa: AK101

    def rec_exp_json_code(self, file, exec_code, ind, count, dec, keys, init_code, con):
        if isinstance(file, (str, list)):
            file = {"type": file}

        if file["type"] == "null":
            aform = ak.forms.IndexedOptionForm(
                "i64",
                ak.forms.EmptyForm(form_key=f"node{count+1}"),
                form_key=f"node{count}",
            )
            dec.append(f"output node{count+1}-data uint8 \n")
            dec.append(f"output node{count}-index int64 \n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-index")
            exec_code.append("\n" + "    " * ind + f"-1 node{count}-index <- stack")
            exec_code.append("\n" + "    " * ind + f"0 node{count+1}-data <- stack")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "record":
            temp = count
            aformcont = []
            aformfields = []
            for elem in file["fields"]:
                aformfields.append(elem["name"])
                (
                    aform,
                    exec_code,
                    count,
                    dec,
                    keys,
                    init_code,
                    con,
                ) = self.rec_exp_json_code(
                    elem, exec_code, ind, count + 1, dec, keys, init_code, con
                )
                aformcont.append(aform)

            aform = ak.forms.RecordForm(aformcont, aformfields, form_key=f"node{temp}")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "string":
            aform = ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm(
                    "uint8", parameters={"__array__": "char"}, form_key=f"node{count+1}"
                ),
                form_key=f"node{count}",
            )
            dec.append(f"output node{count+1}-data uint8 \n")
            dec.append(f"output node{count}-offsets int64 \n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-offsets")
            init_code.append(f"0 node{count}-offsets <- stack\n")

            if self.is_primitive:
                exec_code.append("\n" + "    " * ind + "0 do")
            exec_code.append("\n" + "    " * ind + "stream zigzag-> stack\n")
            exec_code.append(
                "\n" + "    " * ind + f"dup node{count}-offsets +<- stack\n"
            )
            exec_code.append(
                "\n" + "    " * (ind + 1) + f"stream #B-> node{count+1}-data"
            )

            if self.is_primitive:
                exec_code.append("\n" + "    " * ind + "loop")

            return aform, exec_code, count + 1, dec, keys, init_code, con

        elif file["type"] == "int":
            aform = ak.forms.NumpyForm(primitive="int32", form_key=f"node{count}")
            dec.append(f"output node{count}-data int32 \n")
            keys.append(f"node{count}-data")

            if self.is_primitive:
                exec_code.append(
                    "\n" + "    " * ind + f"stream #zigzag-> node{count}-data"
                )
            else:
                exec_code.append(
                    "\n" + "    " * ind + f"stream zigzag-> node{count}-data"
                )

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "long":
            aform = ak.forms.NumpyForm("int64", form_key=f"node{count}")
            keys.append(f"node{count}-data")
            dec.append(f"output node{count}-data int64 \n")

            if self.is_primitive:
                exec_code.append(
                    "\n" + "    " * ind + f"stream #zigzag-> node{count}-data"
                )
            else:
                exec_code.append(
                    "\n" + "    " * ind + f"stream zigzag-> node{count}-data"
                )

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "float":
            aform = ak.forms.NumpyForm("float32", form_key=f"node{count}")
            dec.append(f"output node{count}-data float32 \n")
            keys.append(f"node{count}-data")

            if self.is_primitive:
                exec_code.append("\n" + "    " * ind + f"stream #f-> node{count}-data")
            else:
                exec_code.append("\n" + "    " * ind + f"stream f-> node{count}-data")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "double":
            aform = ak.forms.NumpyForm("float64", form_key=f"node{count}")
            dec.append(f"output node{count}-data float64 \n")
            keys.append(f"node{count}-data")

            if self.is_primitive:
                exec_code.append("\n" + "    " * ind + f"stream #d-> node{count}-data")
            else:
                exec_code.append("\n" + "    " * ind + f"stream d-> node{count}-data")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "boolean":
            aform = ak.forms.NumpyForm("bool", form_key=f"node{count}")
            dec.append(f"output node{count}-data bool\n")
            keys.append(f"node{count}-data")

            if self.is_primitive:
                exec_code.append("\n" + "    " * ind + f"stream #?-> node{count}-data")
            else:
                exec_code.append("\n" + "    " * ind + f"stream ?-> node{count}-data")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "bytes":
            dec.append(f"output node{count+1}-data uint8\n")
            dec.append(f"output node{count}-offsets int64\n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-offsets")
            aform = ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm(
                    "uint8", form_key=f"node{count+1}", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
                form_key=f"node{count}",
            )

            init_code.append(f"0 node{count}-offsets <- stack\n")
            exec_code.append("\n" + "    " * ind + "stream zigzag-> stack\n")
            exec_code.append(
                "\n" + "    " * ind + f"dup node{count}-offsets +<- stack\n"
            )
            exec_code.append(
                "\n" + "    " * (ind + 1) + f"stream #B-> node{count+1}-data"
            )

            return aform, exec_code, count + 1, dec, keys, init_code, con

        elif isinstance(file["type"], list):
            flag = 0
            type_idx = ""
            temp = count
            null_present = False
            out = len(file["type"])
            for elem in file["type"]:
                if isinstance(elem, dict) and elem["type"] == "record":
                    flag = 1
                else:
                    flag = 0

            if "null" in file["type"] and flag == 0 and out == 2:
                dec.append(f"output node{count}-mask int8\n")
                keys.append(f"node{count}-mask")
                type_idx = "null_non_record"

            elif "null" in file["type"] and flag == 1:
                dec.append(f"output node{count}-index int64\n")
                keys.append(f"node{count}-index")
                type_idx = "null_record"

            else:
                for elem in file["type"]:
                    if elem == "null":
                        dec.append(f"output node{count}-mask int8\n")
                        keys.append(f"node{count}-mask")
                        flag = 1
                        mask_idx = count
                        count = count + 1
                        null_present = True
                dec.append(f"output node{count}-tags int8\n")
                keys.append(f"node{count}-tags")
                dec.append(f"output node{count}-index int64\n")
                keys.append(f"node{count}-index")
                union_idx = count
                type_idx = "no_null"

            exec_code.append("\n" + "    " * (ind) + "stream zigzag-> stack case")

            if type_idx == "null_non_record":
                temp = count
                dum_idx = 0
                idxx = file["type"].index("null")
                if out == 2:
                    dum_idx = 1 - idxx
                elif out > 2:
                    if idxx == 0:
                        dum_idx = 1
                    else:
                        dum_idx = idxx - 1

                for i in range(out):
                    if file["type"][i] == "null":
                        if isinstance(file["type"][dum_idx], dict):
                            aa = (
                                "\n"
                                + "    " * (ind + 1)
                                + self.dum_dat(file["type"][dum_idx], temp + 1)
                            )
                        else:
                            aa = (
                                "\n"
                                + "    " * (ind + 1)
                                + self.dum_dat(
                                    {"type": file["type"][dum_idx]}, temp + 1
                                )
                            )
                        exec_code.append(
                            "\n"
                            + "    " * (ind)
                            + f"{i} of 0 node{temp}-mask <- stack {aa} endof"
                        )

                    else:
                        exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of 1 node{temp}-mask <- stack"
                        )

                        (
                            aform1,
                            exec_code,
                            count,
                            dec,
                            keys,
                            init_code,
                            con,
                        ) = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            count + 1,
                            dec,
                            keys,
                            init_code,
                            con,
                        )
                        exec_code.append(" endof")

                aform = ak.forms.ByteMaskedForm(
                    "i8", aform1, True, form_key=f"node{temp}"
                )

            if type_idx == "null_record":
                temp = count
                idxx = file["type"].index("null")
                if out == 2:
                    dum_idx = 1 - idxx
                elif out > 2:
                    if idxx == 0:
                        dum_idx = 1
                    else:
                        dum_idx = 0

                idxx = file["type"].index("null")
                for i in range(out):
                    if file["type"][i] == "null":
                        exec_code.append(
                            "\n"
                            + "    " * (ind)
                            + f"{i} of -1 node{temp}-index <- stack endof"
                        )
                    else:
                        exec_code.append(
                            "\n"
                            + "    " * (ind)
                            + f"{i} of countvar{count}{i} @ node{count}-index <- stack 1 countvar{count}{i} +! "
                        )
                        init_code.append(f"variable countvar{count}{i}\n")
                        (
                            aform1,
                            exec_code,
                            count,
                            dec,
                            keys,
                            init_code,
                            con,
                        ) = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            count + 1,
                            dec,
                            keys,
                            init_code,
                            con,
                        )
                        exec_code.append("\nendof")

                aform = ak.forms.IndexedOptionForm(
                    "i64", aform1, form_key=f"node{temp}"
                )

            if type_idx == "no_null":
                if null_present:
                    idxx = file["type"].index("null")
                if out == 2:
                    dum_idx = 1 - idxx
                elif out > 2:
                    if idxx == 0:
                        dum_idx = 1
                    else:
                        dum_idx = idxx - 1

                temp = count
                temp_forms = []
                for i in range(out):
                    if file["type"][i] == "null":
                        exec_code.append(
                            "\n"
                            + "    " * (ind)
                            + f"{i} of 0 node{mask_idx}-mask <- stack 0 node{union_idx}-tags <- stack 0 node{union_idx}-index <- stack endof"
                        )
                    else:
                        if null_present:
                            exec_code.append(
                                "\n"
                                + "    " * (ind)
                                + f"{i} of 1 node{mask_idx}-mask <- stack {i} node{union_idx}-tags <- stack"
                            )
                        else:
                            exec_code.append(
                                "\n"
                                + "    " * (ind)
                                + f"{i} of {i} node{union_idx}-tags <- stack 1 countvar{count}{i} +!"
                            )
                        init_code.append(f"variable countvar{count}{i} \n")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"countvar{count}{i} @ node{union_idx}-index <- stack"
                        )
                        exec_code.append(
                            "\n" + "    " * (ind + 1) + f"1 countvar{count}{i} +!"
                        )
                        (
                            aform1,
                            exec_code,
                            count,
                            dec,
                            keys,
                            init_code,
                            con,
                        ) = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            count + 1,
                            dec,
                            keys,
                            init_code,
                            con,
                        )
                        temp_forms.append(aform1)
                        exec_code.append("\n endof")

                if null_present:
                    aform = ak.forms.ByteMaskedForm(
                        "i8",
                        ak.forms.UnionForm(
                            "i8", "i64", temp_forms, form_key=f"node{union_idx}"
                        ),
                        True,
                        form_key=f"node{mask_idx}",
                    )
                else:
                    aform = ak.forms.UnionForm(
                        "i8", "i64", aform1, form_key=f"node{mask_idx}"
                    )

            exec_code.append("\n" + "    " * (ind + 1) + "endcase")

            return aform, exec_code, count, dec, keys, init_code, con

        elif isinstance(file["type"], dict):
            (
                aform,
                exec_code,
                count,
                dec,
                keys,
                init_code,
                con,
            ) = self.rec_exp_json_code(
                file["type"], exec_code, ind, count, dec, keys, init_code, con
            )

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "fixed":
            keys.append(f"node{count+1}-data")
            dec.append(f"output node{count+1}-data uint8 \n")
            aform = ak.forms.RegularForm(
                ak.forms.NumpyForm(
                    "uint8", form_key=f"node{count+1}", parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
                size=file["size"],
                form_key=f"node{count}",
            )

            temp = file["size"]
            exec_code.append(
                "\n" + "    " * ind + f"{temp} stream #B-> node{count+1}-data"
            )

            return aform, exec_code, count + 1, dec, keys, init_code, con

        elif file["type"] == "enum":
            aform = ak.forms.IndexedForm(
                "i64",
                ak.forms.ListOffsetForm(
                    "i64",
                    ak.forms.NumpyForm(
                        "uint8",
                        parameters={"__array__": "char"},
                        form_key=f"node{count+2}",
                    ),
                    form_key=f"node{count+1}",
                ),
                parameters={"__array__": "categorical"},
                form_key=f"node{count}",
            )

            keys.append(f"node{count}-index")
            dec.append(f"output node{count}-index int64 \n")

            tempar = file["symbols"]
            offset, dat = [0], []
            prev = 0
            for x in tempar:
                offset.append(len(x) + prev)
                prev = offset[-1]
                for elem in x:
                    dat.append(np.uint8(ord(elem)))

            con[f"node{count+1}-offsets"] = np.array(offset, dtype=np.int64)
            con[f"node{count+2}-data"] = np.array(dat, dtype=np.uint8)
            exec_code.append("\n" + "    " * ind + f"stream zigzag-> node{count}-index")

            return aform, exec_code, count + 2, dec, keys, init_code, con

        elif file["type"] == "array":
            temp = count
            dec.append(f"output node{count}-offsets int64\n")
            init_code.append(f"0 node{count}-offsets <- stack\n")
            exec_code.append("\n" + "    " * ind + "stream zigzag-> stack")
            exec_code.append("\n" + "    " * ind + "dup 0 <")
            exec_code.append(
                "\n" + "    " * ind + "if stream zigzag-> stack drop negate then"
            )
            exec_code.append("\n" + "    " * ind + f"dup node{count}-offsets +<- stack")

            if isinstance(file["items"], str):
                self.is_primitive = True
            else:
                exec_code.append("\n" + "    " * ind + "0 do")

            keys.append(f"node{count}-offsets")
            (
                aformtemp,
                exec_code,
                count,
                dec,
                keys,
                init_code,
                con,
            ) = self.rec_exp_json_code(
                {"type": file["items"]},
                exec_code,
                ind + 1,
                count + 1,
                dec,
                keys,
                init_code,
                con,
            )

            if self.is_primitive:
                self.is_primitive = False
            else:
                exec_code.append("\n" + "    " * ind + "loop")

            exec_code.append("\n" + "    " * ind + "1 stream skip")
            aform = ak.forms.ListOffsetForm("i64", aformtemp, form_key=f"node{temp}")

            return aform, exec_code, count, dec, keys, init_code, con

        elif file["type"] == "map":
            # print(file["name"])
            #         exec_code.append("\npos, inn = decode_varint(pos,fields)"
            #         exec_code.append("\nout = abs(decode_zigzag(inn))"
            #         exec_code.append("\nprint(\"length\",out)"
            #         exec_code.append("\nfor i in range(out):"
            #         exec_code.append("\n"+"    "*(ind+1)+"print(\"{{\")"
            #         exec_code = exec_code+aa
            #         exec_code = exec_code+bb
            #         exec_code = exec_code+ccfa
            #         exec_code = exec_code+dd
            #         exec_code = exec_code+ee
            #         pos,exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,exec_code,ind+1, aform,count+1)
            #         exec_code.append("\n    "*(ind+1)+"print(\":\")"
            #         exec_code = exec_code+ff
            #         pos,exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,exec_code,ind+1, aform,count+1)
            #         exec_code.append("\n"+"    "*(ind+1)+"print(\"}}\")"
            #         exec_code.append("\n"+"    "*ind+"pos, inn = decode_varint(pos,fields)"
            #         jj = "\n"+"    "*ind+"out = decode_zigzag(inn)"
            #         kk = "\n"+"    "*ind+'''if out != 0:
            #     raise
            #             '''
            #         exec_code = exec_code+gg
            #         exec_code = exec_code+hh
            #         exec_code = exec_code+jj
            #         exec_code = exec_code+kk
            raise ak._v2._util.error(NotImplementedError)
