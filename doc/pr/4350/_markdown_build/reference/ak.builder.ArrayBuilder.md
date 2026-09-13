# ak._ext.ArrayBuilder

The low-level ArrayBuilder that builds [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) arrays. This
object is wrapped by [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder).

(Method names in the high-level interface have been changed to include
underscores after “begin” and “end,” but that hasn’t happened in the
low-level interface, yet or possibly at all.)

### *class* ak.layout.ArrayBuilder(initial=1024, resize=8)

#### ArrayBuilder.\_\_getitem_\_(where)

#### ArrayBuilder.\_\_init_\_(initial=1024, resize=8)

#### ArrayBuilder.\_\_iter_\_()

#### ArrayBuilder.\_\_len_\_()

#### ArrayBuilder.\_\_repr_\_()

#### ArrayBuilder.beginlist()

#### ArrayBuilder.beginrecord(name=None)

#### ArrayBuilder.begintuple(arg0)

#### ArrayBuilder.boolean(arg0)

#### ArrayBuilder.bytestring(arg0)

#### ArrayBuilder.clear()

#### ArrayBuilder.endlist()

#### ArrayBuilder.endrecord()

#### ArrayBuilder.endtuple()

#### ArrayBuilder.field(arg0)

#### ArrayBuilder.fromiter(arg0)

#### ArrayBuilder.index(arg0)

#### ArrayBuilder.integer(arg0)

#### ArrayBuilder.null()

#### ArrayBuilder.real(arg0)

#### ArrayBuilder.snapshot()

#### ArrayBuilder.string(arg0)

#### ArrayBuilder.type(arg0)
