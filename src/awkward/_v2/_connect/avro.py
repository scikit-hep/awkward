import numpy as np
import awkward._v2 as ak
import awkward.forth
import json

# from awkward.operations.describe import parameters


class read_avro_py:
    def __init__(self, file_name, show_code=False):
        self.file_name = file_name
        self.data = np.memmap(file_name, np.uint8)
        self.field = []
        pos, self.pairs = self.decode_varint(4, self.data)
        self.pairs = self.decode_zigzag(self.pairs)
        pos = self.cont_spec(pos)
        pos = self.decode_block(pos)
        ind = 2
        exec_code = [
            f'data = np.memmap("{self.file_name}", np.uint8)\npos=0\n',
            """def decode_varint(pos, data):\n\tshift = 0\n\tresult = 0\n\twhile True:\n\t\ti = data[pos]\n\t\tpos += 1\n\t\tresult |= (i & 0x7f) << shift\n\t\tshift += 7\n\t\tif not (i & 0x80):\n\t\t\tbreak\n\treturn pos, result\ndef decode_zigzag(n):\n\treturn (n >> 1) ^ (-(n & 1))\n\n""",
            f"field = {str(self.field)}\n\n",
            f"for i in range({len(self.field)}):\n",
            "    fields = data[field[i][0]:field[i][1]]\n",
            f"    for i in range({sum(self.blocks)}):",
        ]
        aform = []
        dec = ["import struct\nimport numpy \n", "con = {"]
        self.form, self.exec_code, self.count, dec = self.rec_exp_json_code(
            self.schema, exec_code, ind, aform, 0, dec
        )
        dec.append("}")
        dec.append("\n")
        self.head = "".join(dec)
        self.body = "".join(self.exec_code)
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
        self.form = json.loads(temp_json)
        con = loc["con"]
        for key in con.keys():
            con[key] = np.array(con[key])
        self.outarr = ak.from_buffers(
            self.form, sum(self.blocks), con
        )  # put in ._v2 later

    def decode_block(self, pos):
        self.blocks = []
        while True:
            # The byte is random at this position
            # print(self.data.tobytes(), "srgrg")
            # print(pos)
            # print(self.data[temp:].tobytes(), "srgr")
            # while temp < len(self.data)-temp:
            #    print("segeg")
            #    pos, info = self.decode_varint(temp, self.data)
            #    info = self.decode_zigzag(info)
            #    print(info, "srgwrg")
            pos, info = self.decode_varint(pos + 17, self.data)
            info = self.decode_zigzag(info)
            self.blocks.append(int(info))
            # print(self.blocks, "tjhtyrjh")
            # print(pos-temp, "sgrghrg")
            pos, info = self.decode_varint(pos, self.data)
            info = self.decode_zigzag(info)
            # print(pos,info)
            self.field.append([pos, pos + int(info)])
            pos = pos + info
            if pos + 16 == len(self.data):
                break
        return pos

    def cont_spec(self, pos):
        aa = -1
        while aa != 2:
            pos, dat = self.decode_varint(pos, self.data)
            dat = self.decode_zigzag(dat)
            val = self.ret_str(pos, pos + int(dat))
            pos = pos + int(dat)
            aa = self.check_special(val)
            if aa == 0:
                pos, dat = self.decode_varint(pos, self.data)
                dat = self.decode_zigzag(dat)
                val = self.ret_str(pos, pos + int(dat))
                self.codec = val
                pos = pos + int(dat)
            elif aa == 1:

                pos, dat = self.decode_varint(pos, self.data)
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
        return self.data[start:stop].tobytes().decode(errors="surrogateescape")

    def decode_varint(self, pos, _data):
        shift = 0
        result = 0
        while True:
            i = self.data[pos]
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

    def rec_exp_json_code(self, file, exec_code, ind, aform, count, dec):
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
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-index'].append(-1)"
            )
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count+1}-data'].append(np.uint8(0))"
            )
            return aform, exec_code, count, dec

        elif file["type"] == "record":
            # print(file["type"],count)
            temp = count
            aform.append('{"class" : "RecordArray",')
            aform.append('"contents": {\n')
            for elem in file["fields"]:
                aform.append('"' + elem["name"] + '"' + ": ")
                aform, exec_code, count, dec = self.rec_exp_json_code(
                    elem, exec_code, ind, aform, count + 1, dec
                )
            aform[-1] = aform[-1][:-2]
            aform.append(f'}},"form_key": "node{temp}"}},\n')
            return aform, exec_code, count, dec

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
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(np.uint8(out+con['node{count}-offsets'][-1]))"
            )
            # print(pos,abs(out))
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count+1}-data'].extend(fields[pos:pos+abs(out)])"
            )
            exec_code.append(
                "\n"
                + "    " * ind
                + 'print(fields[pos:pos+abs(out)].tobytes().decode(errors="surrogateescape") )'
            )
            exec_code.append("\n" + "    " * ind + "pos = pos + abs(out)")
            return aform, exec_code, count + 1, dec

        elif file["type"] == "int":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "int32", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-data'].append(np.int32(out))"
            )
            exec_code.append("\n" + "    " * ind + "print(out)")
            return aform, exec_code, count, dec

        elif file["type"] == "long":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "int64", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append("\n" + "    " * ind + "print(out)")
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-data'].append(np.int64(out))"
            )
            return aform, exec_code, count, dec

        elif file["type"] == "float":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "float32", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            exec_code.append(
                "\n"
                + "    " * ind
                + 'print(struct.Struct("<f").unpack(fields[pos:pos+4].tobytes())[0])'
            )
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-data'].append(np.float32(struct.Struct(\"<f\").unpack(fields[pos:pos+4].tobytes())[0]))"
            )
            exec_code.append("\n" + "    " * ind + "pos = pos+4")
            return aform, exec_code, count, dec

        elif file["type"] == "double":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "float64", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            exec_code.append(
                "\n"
                + "    " * ind
                + 'print(struct.Struct("<d").unpack(fields[pos:pos+8].tobytes())[0])'
            )
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-data'].append(np.float64(struct.Struct(\"<d\").unpack(fields[pos:pos+8].tobytes())[0]))"
            )
            exec_code.append("\n" + "    " * ind + "pos = pos+8")
            return aform, exec_code, count, dec

        elif file["type"] == "boolean":
            # print(file["name"])
            # print(file["type"],count)
            aform.append(
                f'{{"class" : "NumpyArray", "primitive": "bool", "form_key": "node{count}"}},\n'
            )
            var1 = f" 'node{count}-data'"
            dec.append(var1)
            dec.append(": [],")
            exec_code.append("\n" + "    " * ind + "print(fields[pos])")
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-data'].append(fields[pos])"
            )
            exec_code.append("\n" + "    " * ind + "pos = pos+1")
            return aform, exec_code, count, dec

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
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(out+con['node{count}-offsets'][-1])"
            )
            exec_code.append(
                "\n" + "    " * ind + "print(fields[pos:pos+out].tobytes())"
            )
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count+1}-data'].extend(fields[pos:pos+out])"
            )
            exec_code.append("\n" + "    " * ind + "pos = pos+out")
            return aform, exec_code, count + 1, dec

        elif isinstance(file["type"], list):
            # print(file["name"])
            type_idx = 0
            temp = count
            null_present = False
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "idxx = abs(decode_zigzag(inn))")
            exec_code.append("\n" + "    " * ind + 'print("index :",idxx)')
            out = len(file["type"])
            for elem in file["type"]:
                if isinstance(elem, dict) and elem["type"] == "record":
                    flag = 1
                else:
                    flag = 0
            if "null" in file["type"] and flag == 0 and out == 2:

                aform.append('{"class": "ByteMaskedArray","mask": "i8","content":\n')
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
                        exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{temp}-mask'].append(np.int8(False))"
                        )
                        if isinstance(file["type"][dum_idx], dict):

                            exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                # change dum_dat function to return full string
                                + self.dum_dat(file["type"][dum_idx], temp + 1)
                            )
                        else:
                            exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                # change dum_dat function to return full string
                                + self.dum_dat(
                                    {"type": file["type"][dum_idx]}, temp + 1
                                )
                            )
                    else:
                        exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{count}-mask'].append(np.int8(True))"
                        )

                        aform, exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                aform.append(f'"valid_when": true,"form_key": "node{temp}"}}\n')
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
                        exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{temp}-index'].append(np.int8(-1))"
                        )
                        # exec_code.append(
                        #    "\n"
                        #    + "    " * (ind + 1)
                        # change dum_dat function to return full string
                        #    + self.dum_dat({"type": file["type"][1 - idxx]}, temp + 1)
                        # )
                    else:
                        exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                        exec_code.insert(3, f"countvar{count}{i} = 0\n")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{count}-index'].append(countvar{count}{i})"
                        )
                        exec_code.append(
                            "\n" + "    " * (ind + 1) + f"countvar{count}{i} += 1"
                        )
                        aform, exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                aform.append(f'"valid_when": true,"form_key": "node{temp}"}}\n')
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
                        exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{mask_idx}-mask'].append(np.int8(False))"
                        )
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-tags'].append(np.int8(0))"
                        )
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-index'].append(0)"
                        )
                        # exec_code.append(
                        #    "\n"
                        #    + "    " * (ind + 1)
                        # change dum_dat function to return full string
                        #    + self.dum_dat({"type": file["type"][1 - idxx]}, temp + 1)
                        # )
                    else:
                        if null_present:
                            exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                            exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{mask_idx}-mask'].append(np.int8(True))"
                            )
                            exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{union_idx}-tags'].append(np.int8(idxx))"
                            )
                        else:
                            exec_code.append("\n" + "    " * (ind) + f"if idxx == {i}:")
                            exec_code.append(
                                "\n"
                                + "    " * (ind + 1)
                                + f"con['node{union_idx}-tags'].append(np.int8(idxx))"
                            )
                        exec_code.insert(3, f"countvar{count}{i} = 0\n")
                        exec_code.append(
                            "\n"
                            + "    " * (ind + 1)
                            + f"con['node{union_idx}-index'].append(countvar{count}{i})"
                        )
                        exec_code.append(
                            "\n" + "    " * (ind + 1) + f"countvar{count}{i} += 1"
                        )
                        aform, exec_code, count, dec = self.rec_exp_json_code(
                            {"type": file["type"][i]},
                            exec_code,
                            ind + 1,
                            aform,
                            count + 1,
                            dec,
                        )
                if aform[-1][-2] == ",":
                    aform[-1] = aform[-1][0:-2]
                aform.append(f'], "form_key": "node{union_idx}"}},')
                if null_present:
                    aform.append(f'"valid_when": true,"form_key": "node{mask_idx}"}}\n')
            return aform, exec_code, count, dec

        elif isinstance(file["type"], dict):
            # print(file["name"])
            # print(file["type"],count)
            aform, exec_code, count, dec = self.rec_exp_json_code(
                file["type"], exec_code, ind, aform, count, dec
            )
            return aform, exec_code, count, dec

        elif file["type"] == "fixed":
            # print(file["type"],count)
            var1 = f" 'node{count+1}-data'"
            dec.append(var1)
            dec.append(": [],")
            aform.append(
                f'{{"class": "RegularArray","content": {{"class": "NumpyArray","primitive": "uint8","form_key": "node{count+1}", "parameters": {{"__array__": "byte"}}}},"size": {file["size"]},"form_key": "node{count}"}},\n'
            )
            temp = file["size"]
            exec_code.append("\n" + "    " * ind + f"lenn = {temp}")
            exec_code.append(
                "\n" + "    " * ind + "print(fields[pos:pos+lenn].tobytes())"
            )
            exec_code.append("\n" + "    " * ind + "for i in range(lenn):")
            exec_code.append(
                "\n"
                + "    " * (ind + 1)
                + f"con['node{count+1}-data'].append(np.uint8(fields[i]))"
            )
            exec_code.append("\n" + "    " * ind + "pos = pos+lenn")
            return aform, exec_code, count + 1, dec

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
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append(
                "\n" + "    " * ind + f"con['node{count}-index'].append(out)"
            )
            exec_code.append("\n" + "    " * ind + "print(out)")
            return aform, exec_code, count + 2, dec
        # lark.Tree("indexed", [lark.Tree("listoffset", [lark.Tree("listoffset", [lark.Tree("numpy", [lark.Token("ESCAPED_STRING", '"u1"')]), lark.Tree("is_string", [])])]), lark.Tree("is_categorical", [])])

        elif file["type"] == "array":
            # print(file["name"])
            temp = count
            var1 = f" 'node{count}-offsets'"
            dec.append(var1)
            dec.append(": [0],")
            aform.append('{"class": "ListOffsetArray64","offsets": "i64","content": ')
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append("\n" + "    " * ind + 'print("length",out)')
            exec_code.append(
                "\n"
                + "    " * ind
                + f"con['node{count}-offsets'].append(out+con['node{count}-offsets'][-1])"
            )
            exec_code.append("\n" + "    " * ind + "if out < 0:")
            exec_code.append(
                "\n" + "    " * (ind + 1) + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * (ind + 1) + "nbytes = decode_zigzag(inn)")
            exec_code.append("\n" + "    " * ind + "for j in range(out):")
            aform, exec_code, count, dec = self.rec_exp_json_code(
                {"type": file["items"]}, exec_code, ind + 1, aform, count + 1, dec
            )
            exec_code.append(
                "\n" + "    " * ind + "pos, inn = decode_varint(pos,fields)"
            )
            exec_code.append("\n" + "    " * ind + "out = decode_zigzag(inn)")
            exec_code.append("\n" + "    " * ind + """if out != 0:""")
            exec_code.append("\n" + "    " * (ind + 1) + "raise")
            aform.append(f'"form_key": "node{temp}"}},\n')
            return aform, exec_code, count, dec

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
            raise NotImplementedError


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
                    raise TypeError("File is not a valid avro.")
                pos = 4
                pos, self.pairs = self.decode_varint(4, self.temp_header)
                self.pairs = self.decode_zigzag(self.pairs)
                if self.pairs < 0:
                    pos, self.header_size = self.decode_varint(pos, self.temp_header)
                    self.header_size = self.decode_zigzag(self.pairs)
                    self.pairs = abs(self.pairs)
                pos = self.cont_spec(pos)
                break

            except _ReachedEndofArrayError:
                numbytes *= 2
        ind = 2
        self.update_pos(pos)
        exec_code = []
        init_code = [": init-out\n"]
        header_code = "input stream \n"
        keys = []
        dec = []
        con = {}
        (
            self.form,
            self.exec_code,
            self.count,
            dec,
            keys,
            init_code,
            con,
        ) = self.rec_exp_json_code(
            self.metadata["avro.schema"], exec_code, ind, 0, dec, keys, init_code, con
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
            print(forth_code)  # noqa
        machine = awkward.forth.ForthMachine64(forth_code)
        while True:
            try:

                pos, num_items, len_block = self.decode_block()
                temp_data = self.data.read(len_block)
                if len(temp_data) < len_block:
                    raise _ReachedEndofArrayError
                self.update_pos(len_block)
            except _ReachedEndofArrayError:
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
                raise _ReachedEndofArrayError
            pos, dat = self.decode_varint(pos, self.temp_header)
            dat = self.decode_zigzag(dat)
            val = self.temp_header[pos : pos + int(dat)]
            pos = pos + int(dat)
            if len(val) < int(dat):
                raise _ReachedEndofArrayError
            if key == b"avro.schema":
                self.metadata[key.decode()] = json.loads(val.decode())
            else:
                self.metadata[key.decode()] = val
            temp_count += 1

        return pos

    def check_valid(self):

        init = self.temp_header[0:4]
        if len(init) < 4:
            raise _ReachedEndofArrayError
        if init == b"Obj\x01":
            return True
        else:
            return False

    def decode_varint(self, pos, _data):

        shift = 0
        result = 0
        while True:
            if pos >= len(_data):
                raise _ReachedEndofArrayError
            i = _data[pos]
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

    def rec_exp_json_code(self, file, exec_code, ind, count, dec, keys, init_code, con):
        if isinstance(file, str) or isinstance(file, list):
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
            exec_code.append("\n" + "    " * (ind) + "stream zigzag-> stack case")

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
            for i in range(len(tempar)):
                offset.append(len(tempar[i]) + prev)
                prev = offset[-1]
                for elem in tempar[i]:
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
            exec_code.append("\n" + "    " * ind + "stream zigzag-> stack")
            exec_code.append("\n" + "    " * ind + "if")
            exec_code.append("\n" + "    " * (ind + 1) + "")  # raise an error
            exec_code.append("\n" + "    " * ind + "then")
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
            raise NotImplementedError
