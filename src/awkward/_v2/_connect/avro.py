import numpy as np
import awkward._v2 as ak
import awkward.forth
import json

from awkward.operations.describe import parameters


class read_avro_py:
    def __init__(self, file_name, show_code=False):
        self.file_name = file_name
        self._data = np.memmap(file_name, np.uint8)
        self.field = []
        # try:
        #    if self.check_valid != True:
        #        raise
        # except Exception as error:
        #    raise TypeError("Not a valid avro." + repr(error))
        pos, self.pairs = self.decode_varint(4, self._data)
        self.pairs = self.decode_zigzag(self.pairs)
        print(self.pairs)
        pos = self.cont_spec(pos)
        pos = self.decode_block(pos)
        ind = 2
        _exec_code = [
            f'data = np.memmap("{self.file_name}", np.uint8)\npos=0\n',
            """def decode_varint(pos, data):\n\tshift = 0\n\tresult = 0\n\twhile True:\n\t\ti = data[pos]\n\t\tpos += 1\n\t\tresult |= (i & 0x7f) << shift\n\t\tshift += 7\n\t\tif not (i & 0x80):\n\t\t\tbreak\n\treturn pos, result\ndef decode_zigzag(n):\n\treturn (n >> 1) ^ (-(n & 1))\n\n""",
            f"field = {str(self.field)}\n\n",
            f"for i in range({len(self.field)}):\n",
            "    fields = data[field[i][0]:field[i][1]]\n",
            f"    for i in range({sum(self.blocks)}):",
        ]
        aform = []
        dec = ["import struct\nimport numpy \n", "con = {"]
        self.form, self._exec_code, self.count, dec = self.rec_exp_json_code(
            self.schema, _exec_code, ind, aform, 0, dec
        )
        dec.append("}")
        dec.append("\n")
        self.head = "".join(dec)
        self.body = "".join(self._exec_code)
        # print("".join(head+body))
        loc = {}
        if show_code:
            print("".join(self.head + self.body))  # noqa
        exec("".join(self.head + self.body), globals(), loc)
        # print("".join(self.form)[:-2])
        temp_json = "".join(self.form)[:-2]
        if temp_json[-1] != "}":
            if temp_json[-1] != '"':
                temp_json += '"'
            temp_json += "}"
        print(temp_json)
        self.form = json.loads(temp_json)
        con = loc["con"]
        for key in con.keys():
            con[key] = np.array(con[key])
        print(con)
        self.outarr = ak.from_buffers(
            self.form, sum(self.blocks), con
        )  # put in ._v2 later
        print(self.outarr.layout)

    def decode_block(self, pos):
        self.blocks = []
        while True:
            # The byte is random at this position
            # print(self._data.tobytes(), "srgrg")
            # print(pos)
            temp = pos
            # print(self._data[temp:].tobytes(), "srgr")
            # while temp < len(self._data)-temp:
            #    print("segeg")
            #    pos, info = self.decode_varint(temp, self._data)
            #    info = self.decode_zigzag(info)
            #    print(info, "srgwrg")
            pos, info = self.decode_varint(pos + 17, self._data)
            info = self.decode_zigzag(info)
            self.blocks.append(int(info))
            # print(self.blocks, "tjhtyrjh")
            # print(pos-temp, "sgrghrg")
            pos, info = self.decode_varint(pos, self._data)
            info = self.decode_zigzag(info)
            # print(pos,info)
            self.field.append([pos, pos + int(info)])
            pos = pos + info
            if pos + 16 == len(self._data):
                break
        return pos

    def cont_spec(self, pos):
        aa = -1
        while aa != 2:
            pos, dat = self.decode_varint(pos, self._data)
            dat = self.decode_zigzag(dat)
            val = self.ret_str(pos, pos + int(dat))
            pos = pos + int(dat)
            aa = self.check_special(val)
            if aa == 0:
                pos, dat = self.decode_varint(pos, self._data)
                dat = self.decode_zigzag(dat)
                val = self.ret_str(pos, pos + int(dat))
                self.codec = val
                pos = pos + int(dat)
            elif aa == 1:

                pos, dat = self.decode_varint(pos, self._data)
                dat = self.decode_zigzag(dat)
                val = self.ret_str(pos, pos + int(dat))
                self.schema = json.loads(val)
                pos = pos + int(dat)
                return pos

    def check_special(self, val):
        if val == "avro.codec":
            return 0
        elif val == "avro.schema":
            return 1
        else:
            return 2

    def check_valid(self):
        init = self.ret_str(0, 4)
        if init == "Obj\x01":
            return True
        else:
            return False

    def ret_str(self, start, stop):
        return self._data[start:stop].tobytes().decode(errors="surrogateescape")

    def decode_varint(self, pos, _data):
        shift = 0
        result = 0
        while True:
            i = self._data[pos]
            pos += 1
            result |= (i & 0x7F) << shift
            shift += 7
            if not (i & 0x80):
                break
        return pos, result

    def decode_zigzag(self, n):
        return (n >> 1) ^ (-(n & 1))

    def dum_dat(self, dtype, count):
        if dtype["type"] == "int":
            return f"con['node{count}-data'].append(np.int32(0))"
        if dtype["type"] == "long":
            return f"con['node{count}-data'].append(np.int64(0))"
        if dtype["type"] == "float":
            return f"con['node{count}-data'].append(np.float32(0))"
        if dtype["type"] == "double":
            return f"con['node{count}-data'].append(np.float64(0))"
        if dtype["type"] == "boolean":
            return f"con['node{count}-data'].append(0)"
        if dtype["type"] == "bytes":
            return (
                f"con['node{count}-offsets'].append(1+con['node{count}-offsets'][-1])"
                + f"con['node{count+1}-data'].extend([b'a']])"
            )
        if dtype["type"] == "string":
            # \ncon['node{count+1}-data'].extend([114])"
            code = f"con['node{count}-offsets'].append(np.uint8(0+con['node{count}-offsets'][-1]))"
            return code
        if dtype["type"] == "enum":
            return f"con['node{count}-index'].append(0)"

    def rec_exp_json_code(self, file, _exec_code, ind, aform, count, dec):
        if isinstance(file, str) or isinstance(file, list):
            file = {"type": file}
        if (
            file["type"] == "null"
        ):  # problem is that a null array does not have any length
            aform.append(
                f'{{"class": "IndexedOptionArray64","index": "i64","content": {{"class": "EmptyArray","form_key": "node{count+1}"}},"form_key": "node{count}"}}'
            )
            var1 = f" 'node{count}-index'"
            var2 = f" 'node{count+1}-data'"
            dec.append(var1)
            dec.append(": [],")
            dec.append(var2)
            dec.append(": [],")
            _exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-index'].append(-1)"
            )
            _exec_code.append(
                "\n" + "    " * ind +
                f"con['node{count+1}-data'].append(np.uint8(0))"
            )
            return aform, _exec_code, count, dec

        elif file["type"] == "record":
            # print(file["type"],count)
            temp = count
            aform.append('{"class" : "RecordArray",')
            aform.append('"contents": {\n')
            for elem in file["fields"]:
                aform.append('"' + elem["name"] + '"' + ": ")
                aform, _exec_code, count, dec = self.rec_exp_json_code(
                    elem, _exec_code, ind, aform, count + 1, dec
                )
            aform[-1] = aform[-1][:-2]
            aform.append(f'}},"form_key": "node{temp}"}},\n')
            return aform, _exec_code, count, dec

        elif file["type"] == "string":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class": "ListOffsetArray64","offsets": "i64","content": {{"class": "NumpyArray","primitive": "uint8","parameters": {{"__array__": "char"}},"form_key": "node{count+1}"}},"parameters": {{"__array__": "string"}},"form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-offsets'"
            var2 = f" 'node{count+1}-data'"
            dec.append(var1)
            dec.append(": [0],")
            dec.append(var2)
            dec.append(": [],")
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(np.uint8(out+con['node{count}-offsets'][-1]))"
            )
            # print(pos,abs(out))
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count+1}-data'].extend(fields[pos:pos+abs(out)])"
            )
            _exec_code.append(
                "\n"
                + "    " * ind
                + 'print(fields[pos:pos+abs(out)].tobytes().decode(errors="surrogateescape") )'
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos + abs(out)")
            return aform, _exec_code, count + 1, dec

        elif file["type"] == "int":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "int32", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append(
                "\n" + "    " * ind +
                f"con['node{count}-data'].append(np.int32(out))"
            )
            _exec_code.append("\n" + "    " * ind + "print(out)")
            return aform, _exec_code, count, dec

        elif file["type"] == "long":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "int64", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append("\n" + "    " * ind + "print(out)")
            _exec_code.append(
                "\n" + "    " * ind +
                f"con['node{count}-data'].append(np.int64(out))"
            )
            return aform, _exec_code, count, dec

        elif file["type"] == "float":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "float32", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            _exec_code.append(
                "\n"
                + "    " * ind
                + 'print(struct.Struct("<f").unpack(fields[pos:pos+4].tobytes())[0])'
            )
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-data'].append(np.float32(struct.Struct(\"<f\").unpack(fields[pos:pos+4].tobytes())[0]))"
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos+4")
            return aform, _exec_code, count, dec

        elif file["type"] == "double":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "float64", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            _exec_code.append(
                "\n"
                + "    " * ind
                + 'print(struct.Struct("<d").unpack(fields[pos:pos+8].tobytes())[0])'
            )
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-data'].append(np.float64(struct.Struct(\"<d\").unpack(fields[pos:pos+8].tobytes())[0]))"
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos+8")
            return aform, _exec_code, count, dec

        elif file["type"] == "boolean":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "bool", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            _exec_code.append("\n" + "    " * ind + "print(fields[pos])")
            _exec_code.append(
                "\n" + "    " * ind +
                f"con['node{count}-data'].append(fields[pos])"
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos+1")
            return aform, _exec_code, count, dec

        elif file["type"] == "bytes":
            # print(file["name"])
            # print(file["type"],count)
            astring = f'{{"class": "ListOffsetArray64","offsets": "i64","content": {{"class": "NumpyArray", "primitive": "uint8","parameters": {{"__array__": "byte"}}, "form_key": "node{count+1}"}},"parameters": {{"__array__": "byte"}},"form_key": "node{count}"}},\n'
            var1 = f" 'node{count+1}-data'"
            var2 = f" 'node{count}-offsets'"
            dec.append(var2)
            dec.append(": [0],")
            dec.append(var1)
            dec.append(": [],")
            aform.append(astring)
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(out+con['node{count}-offsets'][-1])"
            )
            _exec_code.append(
                "\n" + "    " * ind + "print(fields[pos:pos+out].tobytes())"
            )
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count+1}-data'].extend(fields[pos:pos+out])"
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos+out")
            return aform, _exec_code, count + 1, dec

        elif isinstance(file["type"], list):
            # print(file["name"])
            type_idx = 0
            temp = count
            null_present = False
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind +
                              "idxx = abs(decode_zigzag(inn))")
            _exec_code.append("\n" + "    " * ind + 'print("index :",idxx)')
            out = len(file["type"])
            for elem in file["type"]:
                if isinstance(elem, dict) and elem["type"] == "record":
                    flag = 1
                else:
                    flag = 0
            if "null" in file["type"] and flag == 0 and out == 2:

                aform.append(
                    '{"class": "ByteMaskedArray","mask": "i8","content":\n')
                var1 = f" 'node{count}-mask'"
                dec.append(var1)
                dec.append(": [],")
                type_idx = 0
            elif "null" in file["type"] and flag == 1:
                aform.append(
                    '{"class": "IndexedOptionArray64","index": "i64","content":\n'
                )
                var1 = f" 'node{count}-index'"
                dec.append(var1)
                dec.append(": [],")
                type_idx = 1
            else:
                for elem in file["type"]:
                    if elem == "null":
                        aform.append(
                            '{"class": "ByteMaskedArray","mask": "i8","content":\n'
                        )
                        var1 = f" 'node{count}-mask'"
                        mask_idx = count
                        dec.append(var1)
                        dec.append(": [],")
                        count = count + 1
                        null_present = True
                aform.append(
                    '{"class": "UnionArray8_64","tags": "i8","index": "i64","contents": [\n'
                )
                var1 = f" 'node{count}-tags'"
                union_idx = count
                dec.append(var1)
                dec.append(": [],")
                var1 = f" 'node{count}-index'"
                dec.append(var1)
                dec.append(": [],")
                type_idx = 2
            print(type_idx)
            print("GSEGSEGEGEG")
            if type_idx == 0:
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
                    print(file["type"][i])
                    if file["type"][i] == "null":
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"if idxx == {i}:")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{temp}-mask'].append(np.int8(False))"
                        )
                        if isinstance(file["type"][dum_idx], dict):

                            _exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                # change dum_dat function to return full string
                                + self.dum_dat(file["type"][dum_idx], temp + 1)
                            )
                        else:
                            _exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                # change dum_dat function to return full string
                                + self.dum_dat(
                                    {"type": file["type"][dum_idx]}, temp + 1
                                )
                            )
                    else:
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"if idxx == {i}:")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{count}-mask'].append(np.int8(True))"
                        )

                        aform, _exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                aform.append(
                    f'"valid_when": true,"form_key": "node{temp}"}}\n')
            if type_idx == 1:
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
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"if idxx == {i}:")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{temp}-index'].append(np.int8(-1))"
                        )

                        print({"type": file["type"][dum_idx]})
                        # _exec_code.append(
                        #    "\n"
                        #    + "    " * (ind + 1)
                        # change dum_dat function to return full string
                        #    + self.dum_dat({"type": file["type"][1 - idxx]}, temp + 1)
                        # )
                    else:
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"if idxx == {i}:")
                        _exec_code.insert(3, f"countvar{count}{i} = 0\n")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{count}-index'].append(countvar{count}{i})"
                        )
                        _exec_code.append(
                            "\n" + "    " * (ind + 1) +
                            f"countvar{count}{i} += 1"
                        )
                        aform, _exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                aform.append(
                    f'"valid_when": true,"form_key": "node{temp}"}}\n')
            if type_idx == 2:
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
                # idxx = file["type"].index("null")
                for i in range(out):
                    if file["type"][i] == "null":
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"if idxx == {i}:")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{mask_idx}-mask'].append(np.int8(False))"
                        )
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-tags'].append(np.int8(0))"
                        )
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-index'].append(0)"
                        )
                        print({"type": file["type"][dum_idx]})
                        # _exec_code.append(
                        #    "\n"
                        #    + "    " * (ind + 1)
                        # change dum_dat function to return full string
                        #    + self.dum_dat({"type": file["type"][1 - idxx]}, temp + 1)
                        # )
                    else:
                        if null_present:
                            _exec_code.append(
                                "\n" + "    " * (ind) + f"if idxx == {i}:"
                            )
                            _exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{mask_idx}-mask'].append(np.int8(True))"
                            )
                            _exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{union_idx}-tags'].append(np.int8(idxx))"
                            )
                        else:
                            _exec_code.append(
                                "\n" + "    " * (ind) + f"if idxx == {i}:"
                            )
                            _exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{union_idx}-tags'].append(np.int8(idxx))"
                            )
                        _exec_code.insert(3, f"countvar{count}{i} = 0\n")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-index'].append(countvar{count}{i})"
                        )
                        _exec_code.append(
                            "\n" + "    " * (ind + 1) +
                            f"countvar{count}{i} += 1"
                        )
                        aform, _exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                if aform[-1][-2] == ",":
                    aform[-1] = aform[-1][0:-2]
                aform.append(f'], "form_key": "node{union_idx}"}},')
                if null_present:
                    aform.append(
                        f'"valid_when": true,"form_key": "node{mask_idx}"}}\n')
            return aform, _exec_code, count, dec

        elif isinstance(file["type"], dict):
            # print(file["name"])
            # print(file["type"],count)
            aform, _exec_code, count, dec = self.rec_exp_json_code(
                file["type"], _exec_code, ind, aform, count, dec
            )
            return aform, _exec_code, count, dec

        elif file["type"] == "fixed":
            # print(file["type"],count)
            var1 = f" 'node{count+1}-data'"
            dec.append(var1)
            dec.append(": [],")
            aform.append(
                f'{{"class": "RegularArray","content": {{"class": "NumpyArray","primitive": "uint8","form_key": "node{count+1}", "parameters": {{"__array__": "byte"}}}},"size": {file["size"]},"form_key": "node{count}"}},\n'
            )
            temp = file["size"]
            _exec_code.append("\n" + "    " * ind + f"lenn = {temp}")
            _exec_code.append(
                "\n" + "    " * ind + "print(fields[pos:pos+lenn].tobytes())"
            )
            _exec_code.append("\n" + "    " * ind + "for i in range(lenn):")
            _exec_code.append(
                "\n"
                + "    " * (ind + 1)
                + f"con['node{count+1}-data'].append(np.uint8(fields[i]))"
            )
            _exec_code.append("\n" + "    " * ind + "pos = pos+lenn")
            return aform, _exec_code, count + 1, dec

        elif file["type"] == "enum":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class": "IndexedArray64","index": "i64","content": {{"class": "ListOffsetArray64","offsets": "i64","content": {{"class": "NumpyArray","primitive": "uint8","parameters": {{"__array__": "char"}},"form_key": "node{count+2}"}},"parameters": {{"__array__": "string"}},"form_key": "node{count+1}"}}, "parameters": {{"__array__": "categorical"}}, "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-index'"
            dec.append(var1)
            dec.append(": [],")
            tempar = file["symbols"]
            offset, dat = [0], []
            prev = 0
            for i in range(len(tempar)):
                offset.append(len(tempar[i]) + prev)
                prev = offset[-1]
                for elem in tempar[i]:
                    dat.append(np.uint8(ord(elem)))
            var2 = f" 'node{count+1}-offsets': {str(offset)},"
            dec.append(var2)
            var2 = f" 'node{count+2}-data': np.array({str(dat)},dtype = np.uint8),"
            dec.append(var2)
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-index'].append(out)"
            )
            _exec_code.append("\n" + "    " * ind + "print(out)")
            return aform, _exec_code, count + 2, dec
        # lark.Tree("indexed", [lark.Tree("listoffset", [lark.Tree("listoffset", [lark.Tree("numpy", [lark.Token("ESCAPED_STRING", '"u1"')]), lark.Tree("is_string", [])])]), lark.Tree("is_categorical", [])])

        elif file["type"] == "array":
            # print(file["name"])
            temp = count
            var1 = f" 'node{count}-offsets'"
            dec.append(var1)
            dec.append(": [0],")
            aform.append(
                '{"class": "ListOffsetArray64","offsets": "i64","content": ')
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append("\n" + "    " * ind + 'print("length",out)')
            _exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(out+con['node{count}-offsets'][-1])"
            )
            _exec_code.append("\n" + "    " * ind + "if out < 0:")
            _exec_code.append(
                "\n" + "    " * (ind + 1) +
                "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * (ind + 1) +
                              "nbytes = decode_zigzag(inn)")
            _exec_code.append("\n" + "    " * ind + "for j in range(out):")
            aform, _exec_code, count, dec = self.rec_exp_json_code(
                {"type": file["items"]}, _exec_code, ind +
                1, aform, count + 1, dec
            )
            _exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            _exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            _exec_code.append("\n" + "    " * ind + """if out != 0:""")
            _exec_code.append("\n" + "    " * (ind + 1) + "raise")
            aform.append(f'"form_key": "node{temp}"}},\n')
            return aform, _exec_code, count, dec

        elif file["type"] == "map":
            # print(file["name"])
            #         _exec_code.append("\npos, inn = decode_varint(pos,fields)"
            #         _exec_code.append("\nout = abs(decode_zigzag(inn))"
            #         _exec_code.append("\nprint(\"length\",out)"
            #         _exec_code.append("\nfor i in range(out):"
            #         _exec_code.append("\n"+"    "*(ind+1)+"print(\"{{\")"
            #         _exec_code = _exec_code+aa
            #         _exec_code = _exec_code+bb
            #         _exec_code = _exec_code+ccfa
            #         _exec_code = _exec_code+dd
            #         _exec_code = _exec_code+ee
            #         pos,_exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,_exec_code,ind+1, aform,count+1)
            #         _exec_code.append("\n    "*(ind+1)+"print(\":\")"
            #         _exec_code = _exec_code+ff
            #         pos,_exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,_exec_code,ind+1, aform,count+1)
            #         _exec_code.append("\n"+"    "*(ind+1)+"print(\"}}\")"
            #         _exec_code.append("\n"+"    "*ind+"pos, inn = decode_varint(pos,fields)"
            #         jj = "\n"+"    "*ind+"out = decode_zigzag(inn)"
            #         kk = "\n"+"    "*ind+'''if out != 0:
            #     raise
            #             '''
            #         _exec_code = _exec_code+gg
            #         _exec_code = _exec_code+hh
            #         _exec_code = _exec_code+jj
            #         _exec_code = _exec_code+kk
            raise NotImplementedError


class read_avro_ft:
    def __init__(self, file_name, show_code=False):
        self.file_name = file_name
        self._data = np.memmap(file_name, np.uint8)
        self.field = []
        pos, self.pairs = self.decode_varint(4, self._data)
        self.pairs = self.decode_zigzag(self.pairs)
        pos = self.cont_spec(pos)
        ind = 2
        _exec_code = []
        _init_code = [": init-out\n"]
        header_code = "input stream \n"
        keys = []
        dec = []
        con = {}
        self.form, self._exec_code, self.count, dec, keys, _init_code, con = self.rec_exp_json_code(
            self.schema, _exec_code, ind, 0, dec, keys, _init_code, con
        )
        print(self.form)
        first_iter = True
        _init_code.append(";\n")
        pos, num_items, len_block = self.decode_block(pos)
        header_code = header_code + "".join(dec)
        header_code = header_code+"".join(_init_code)
        _exec_code.insert(0, "0 do \n")
        _exec_code1 = "".join(_exec_code)
        forth_code = header_code+_exec_code1+"\nloop"
        print(forth_code)
        print("ASDFNOEJGOGEON")
        machine = awkward.forth.ForthMachine64(forth_code)
        machine.stack_push(num_items)
        while pos+16 < len(self._data):
            if first_iter:
                machine.begin({"stream": self._data[pos: pos+len_block]})
                print(ak.Array(machine.bytecodes).to_list())
                machine.call("init-out")
                machine.resume()
                first_iter = False
            else:
                pos, num_items, len_block = self.decode_block(pos)
                machine.stack_push(num_items)
                machine.begin_again(
                    {"stream": self._data[pos: pos+len_block]}, True)
                machine.resume()
            pos += len_block

        for elem in keys:
            if "offsets" in elem:
                con[elem] = machine.output_Index64(elem)
            else:
                con[elem] = machine.output_NumpyArray(elem)
        print(con)
        self.outarr = ak.from_buffers(
            self.form, sum(self.blocks), con
        )

    def decode_block(self, pos):
        self.blocks = []
        pos, info = self.decode_varint(pos + 17, self._data)
        info1 = self.decode_zigzag(info)
        self.blocks.append(int(info1))
        pos, info = self.decode_varint(pos, self._data)
        info2 = self.decode_zigzag(info)
        self.field.append([pos, pos + int(info2)])
        return pos, info1, info2

    def cont_spec(self, pos):
        aa = -1
        while aa != 2:
            pos, dat = self.decode_varint(pos, self._data)
            dat = self.decode_zigzag(dat)
            val = self.ret_str(pos, pos + int(dat))
            pos = pos + int(dat)
            aa = self.check_special(val)
            if aa == 0:
                pos, dat = self.decode_varint(pos, self._data)
                dat = self.decode_zigzag(dat)
                val = self.ret_str(pos, pos + int(dat))
                self.codec = val
                pos = pos + int(dat)
            elif aa == 1:

                pos, dat = self.decode_varint(pos, self._data)
                dat = self.decode_zigzag(dat)
                val = self.ret_str(pos, pos + int(dat))
                self.schema = json.loads(val)
                pos = pos + int(dat)
                return pos

    def check_special(self, val):
        if val == "avro.codec":
            return 0
        elif val == "avro.schema":
            return 1
        else:
            return 2

    def check_valid(self):
        init = self.ret_str(0, 4)
        if init == "Obj\x01":
            return True
        else:
            return False

    def ret_str(self, start, stop):
        return self._data[start:stop].tobytes().decode(errors="surrogateescape")

    def decode_varint(self, pos, _data):
        shift = 0
        result = 0
        while True:
            i = self._data[pos]
            pos += 1
            result |= (i & 0x7F) << shift
            shift += 7
            if not (i & 0x80):
                break
        return pos, result

    def decode_zigzag(self, n):
        return (n >> 1) ^ (-(n & 1))

    def dum_dat(self, dtype, count):
        if dtype["type"] == "int":
            return f"0 node{count}-data <- stack "
        if dtype["type"] == "long":
            return f"0 node{count}-data <- stack "
        if dtype["type"] == "float":
            return f"0 node{count}-data <- stack "
        if dtype["type"] == "double":
            return f"0 node{count}-data <- stack "
        if dtype["type"] == "boolean":
            return f"0 node{count}-data <- stack "
        if dtype["type"] == "bytes":
            return f"1 node{count}-offsets +<- stack 97 node{count+1}-data <- stack "
        if dtype["type"] == "string":
            return f"0 node{count}-offsets +<- stack "
        if dtype["type"] == "enum":
            return f"0 node{count}-index <- stack "

    def rec_exp_json_code(self, file, _exec_code, ind, count, dec, keys, _init_code, con):
        if isinstance(file, str) or isinstance(file, list):
            file = {"type": file}
        if (
            file["type"] == "null"
        ):
            aform = ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm(
                form_key=f"node{count+1}"), form_key=f"node{count}")
            dec.append(f"output node{count+1}-data uint8 \n")
            dec.append(f"output node{count}-index int64 \n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-index")
            _exec_code.append("\n" + "    " * ind +
                              f'-1 node{count}-index <- stack')
            _exec_code.append("\n" + "    " * ind +
                              f'0 node{count+1}-data <- stack')
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "record":
            temp = count
            aformcont = []
            aformfields = []
            for elem in file["fields"]:
                aformfields.append(elem["name"])
                aform, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                    elem, _exec_code, ind, count + 1, dec, keys, _init_code, con
                )
                aformcont.append(aform)
            aform = ak.forms.RecordForm(
                aformcont, aformfields, form_key=f"node{temp}")
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "string":
            aform = ak.forms.ListOffsetForm(
                "i64", ak.forms.NumpyForm("uint8", parameters={"__array__": "char"}, form_key=f"node{count+1}"), form_key=f"node{count}")
            dec.append(f"output node{count+1}-data uint8 \n")
            dec.append(f"output node{count}-offsets int64 \n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-offsets")
            _init_code.append(f"0 node{count}-offsets <- stack\n")
            _exec_code.append("\n" + "    " * ind +
                              'stream zigzag-> stack\n')
            _exec_code.append("\n" + "    " * ind +
                              f'dup node{count}-offsets +<- stack\n')
            _exec_code.append("\n" + "    " * ind +
                              '0 do')
            _exec_code.append("\n" + "    " * (ind+1) +
                              f'stream B-> node{count+1}-data')
            _exec_code.append("\n" + "    " * ind +
                              'loop')
            return aform, _exec_code, count + 1, dec, keys, _init_code, con

        elif file["type"] == "int":
            aform = ak.forms.NumpyForm(
                primitive="int32", form_key=f"node{count}")
            dec.append(f"output node{count}-data int32 \n")
            keys.append(f"node{count}-data")
            _exec_code.append("\n" + "    " * ind +
                              f'stream zigzag-> node{count}-data')
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "long":
            aform = ak.forms.NumpyForm("int64", form_key=f"node{count}")
            keys.append(f"node{count}-data")
            dec.append(f"output node{count}-data int64 \n")
            _exec_code.append("\n" + "    " * ind +
                              f'stream zigzag-> node{count}-data')
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "float":
            aform = ak.forms.NumpyForm("float32", form_key=f"node{count}")
            dec.append(f"output node{count}-data float32 \n")
            keys.append(f"node{count}-data")
            _exec_code.append(
                "\n"
                + "    " * ind
                + f'stream f-> node{count}-data'
            )
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "double":
            aform = ak.forms.NumpyForm("float64", form_key=f"node{count}")
            dec.append(f"output node{count}-data float64 \n")
            keys.append(f"node{count}-data")
            _exec_code.append(
                "\n"
                + "    " * ind
                + f'stream d-> node{count}-data'
            )
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "boolean":
            aform = ak.forms.NumpyForm("bool", form_key=f"node{count}")
            dec.append(f"output node{count}-data bool\n")
            keys.append(f"node{count}-data")
            _exec_code.append(
                "\n"
                + "    " * ind
                + f'stream ?-> node{count}-data'
            )
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "bytes":
            dec.append(f"output node{count+1}-data uint8\n")
            dec.append(f"output node{count}-offsets int64\n")
            keys.append(f"node{count+1}-data")
            keys.append(f"node{count}-offsets")
            aform = ak.forms.ListOffsetForm(
                "i64", ak.forms.NumpyForm("uint8", form_key=f"node{count+1}", parameters={"__array__": "byte"}), parameters={"__array__": "bytestring"}, form_key=f"node{count}")

            _init_code.append(f"0 node{count}-offsets <- stack\n")
            _exec_code.append("\n" + "    " * ind +
                              'stream zigzag-> stack\n')
            _exec_code.append("\n" + "    " * ind +
                              f'dup node{count}-offsets +<- stack\n')
            _exec_code.append("\n" + "    " * ind +
                              '0 do')
            _exec_code.append("\n" + "    " * (ind+1) +
                              f'stream B-> node{count+1}-data')
            _exec_code.append("\n" + "    " * ind +
                              'loop')
            return aform, _exec_code, count + 1, dec, keys, _init_code, con

        elif isinstance(file["type"], list):
            flag = 0
            type_idx = 0
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
                type_idx = 0
            elif "null" in file["type"] and flag == 1:
                dec.append(f"output node{count}-index int64\n")
                keys.append(f"node{count}-index")
                type_idx = 1
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
                type_idx = 2
            _exec_code.append(
                "\n"
                + "    " * (ind)
                + "stream zigzag-> stack case"
            )
            if type_idx == 0:
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

                            aa = "\n" + "    " * \
                                (ind + 1) + \
                                self.dum_dat(file["type"][dum_idx], temp + 1)
                        else:
                            aa = "\n" + "    " * \
                                (ind + 1) + \
                                self.dum_dat(
                                    {"type": file["type"][dum_idx]}, temp + 1)
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of 0 node{temp}-mask <- stack {aa} endof")
                    else:
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of 1 node{temp}-mask <- stack")

                        aform1, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            count + 1,
                            dec, keys, _init_code, con
                        )
                        _exec_code.append(" endof")
                aform = ak.forms.ByteMaskedForm(
                    "i8", aform1, True, form_key=f"node{temp}")
            if type_idx == 1:
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
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of -1 node{temp}-index <- stack endof")

                        print({"type": file["type"][dum_idx]})
                    else:
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of countvar{count}{i} @ node{count}-index <- stack 1 countvar{count}{i} +! ")
                        _init_code.append(f"variable countvar{count}{i}\n")
                        aform1, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            count + 1,
                            dec, keys, _init_code, con
                        )
                        _exec_code.append("\nendof")
                aform = ak.forms.IndexedOptionForm(
                    "i64", aform1, form_key=f"node{temp}")
            if type_idx == 2:
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
                        _exec_code.append(
                            "\n" + "    " * (ind) + f"{i} of 0 node{mask_idx}-mask <- stack 0 node{union_idx}-tags <- stack 0 node{union_idx}-index <- stack endof")
                    else:
                        if null_present:
                            _exec_code.append(
                                "\n" + "    " *
                                (ind) +
                                f"{i} of 1 node{mask_idx}-mask <- stack {i} node{union_idx}-tags <- stack"
                            )
                        else:
                            _exec_code.append(
                                "\n" + "    " *
                                (ind) +
                                f"{i} of {i} node{union_idx}-tags <- stack 1 countvar{count}{i} +!"
                            )
                        _init_code.append(f"variable countvar{count}{i} \n")
                        _exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"countvar{count}{i} @ node{union_idx}-index <- stack"
                        )
                        _exec_code.append(
                            "\n" + "    " * (ind + 1) +
                            f"1 countvar{count}{i} +!"
                        )
                        aform1, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            _exec_code,
                            ind + 1,
                            count + 1,
                            dec, keys, _init_code, con
                        )
                        temp_forms.append(aform1)
                        _exec_code.append("\n endof")

                if null_present:
                    aform = ak.forms.ByteMaskedForm(
                        "i8", ak.forms.UnionForm(
                            "i8", "i64", temp_forms, form_key=f"node{union_idx}"), True, form_key=f"node{mask_idx}")
                else:
                    aform = ak.forms.UnionForm(
                        "i8", "i64", aform1, form_key=f"node{mask_idx}")
            _exec_code.append(
                "\n" + "    " * (ind + 1) +
                "endcase"
            )
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif isinstance(file["type"], dict):
            aform, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                file["type"], _exec_code, ind, count, dec, keys, _init_code, con
            )
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "fixed":
            keys.append(f"node{count+1}-data")
            dec.append(f"output node{count+1}-data uint8 \n")
            aform = ak.forms.RegularForm(ak.forms.NumpyForm("uint8", form_key=f"node{count+1}", parameters={
                "__array__": "byte"}), parameters={
                "__array__": "bytestring"}, size=file["size"], form_key=f"node{count}")
            temp = file["size"]
            _exec_code.append(
                "\n" + "    " * ind + f"{temp} stream #B-> node{count+1}-data"
            )
            return aform, _exec_code, count + 1, dec, keys, _init_code, con

        elif file["type"] == "enum":
            aform = ak.forms.IndexedForm("i64", ak.forms.ListOffsetForm(
                "i64", ak.forms.NumpyForm("uint8", parameters={"__array__": "char"}, form_key=f"node{count+2}"), form_key=f"node{count+1}"), parameters={"__array__": "categorical"}, form_key=f"node{count}")
            keys.append(f"node{count}-index")
            dec.append(f"output node{count}-index int64 \n")
            tempar = file["symbols"]
            offset, dat = [0], []
            prev = 0
            for i in range(len(tempar)):
                offset.append(len(tempar[i]) + prev)
                prev = offset[-1]
                for elem in tempar[i]:
                    dat.append(np.uint8(ord(elem)))
            con[f'node{count+1}-offsets'] = np.array(offset, dtype=np.int64)
            con[f'node{count+2}-data'] = np.array(dat, dtype=np.uint8)
            _exec_code.append(
                "\n" + "    " * ind + f"stream zigzag-> node{count}-index"
            )
            return aform, _exec_code, count + 2, dec, keys, _init_code, con
        elif file["type"] == "array":
            temp = count
            dec.append(
                f"output node{count}-offsets int64\n")
            _init_code.append(f"0 node{count}-offsets <- stack\n")
            _exec_code.append(
                "\n" + "    " * ind + "stream zigzag-> stack"
            )
            _exec_code.append(
                "\n" + "    " * ind + f"dup node{count}-offsets +<- stack"
            )
            keys.append(f"node{count}-offsets")
            _exec_code.append("\n" + "    " * ind + "0 do")
            aformtemp, _exec_code, count, dec, keys, _init_code, con = self.rec_exp_json_code(
                {"type": file["items"]}, _exec_code, ind +
                1, count + 1, dec, keys, _init_code, con
            )
            _exec_code.append(
                "\n" + "    " * ind + "loop"
            )
            _exec_code.append(
                "\n" + "    " * ind + "stream zigzag-> stack"
            )
            _exec_code.append("\n" + "    " * ind + "if")
            _exec_code.append("\n" + "    " * (ind+1) +
                              "")  # raise an error
            _exec_code.append("\n" + "    " * ind + "then")
            aform = ak.forms.ListOffsetForm(
                "i64", aformtemp, form_key=f"node{temp}")
            return aform, _exec_code, count, dec, keys, _init_code, con

        elif file["type"] == "map":
            # print(file["name"])
            #         _exec_code.append("\npos, inn = decode_varint(pos,fields)"
            #         _exec_code.append("\nout = abs(decode_zigzag(inn))"
            #         _exec_code.append("\nprint(\"length\",out)"
            #         _exec_code.append("\nfor i in range(out):"
            #         _exec_code.append("\n"+"    "*(ind+1)+"print(\"{{\")"
            #         _exec_code = _exec_code+aa
            #         _exec_code = _exec_code+bb
            #         _exec_code = _exec_code+ccfa
            #         _exec_code = _exec_code+dd
            #         _exec_code = _exec_code+ee
            #         pos,_exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,_exec_code,ind+1, aform,count+1)
            #         _exec_code.append("\n    "*(ind+1)+"print(\":\")"
            #         _exec_code = _exec_code+ff
            #         pos,_exec_code,count = self.rec_exp_json_code({"type": file["values"]},fields,pos,_exec_code,ind+1, aform,count+1)
            #         _exec_code.append("\n"+"    "*(ind+1)+"print(\"}}\")"
            #         _exec_code.append("\n"+"    "*ind+"pos, inn = decode_varint(pos,fields)"
            #         jj = "\n"+"    "*ind+"out = decode_zigzag(inn)"
            #         kk = "\n"+"    "*ind+'''if out != 0:
            #     raise
            #             '''
            #         _exec_code = _exec_code+gg
            #         _exec_code = _exec_code+hh
            #         _exec_code = _exec_code+jj
            #         _exec_code = _exec_code+kk
            raise NotImplementedError
