import fastparquet
import fastparquet.core
import fastparquet.thrift_structures

file = open("/home/jpivarski/storage/data/chep-2019-jagged-jagged-jagged/sample-jagged3-nodict.parquet", "rb")
parquetfile = fastparquet.ParquetFile(file)

for rg in parquetfile.filter_row_groups([]):
    col = rg.columns[0]
    file.seek(col.meta_data.data_page_offset)
    ph = fastparquet.thrift_structures.read_thrift(file, fastparquet.thrift_structures.parquet_thrift.PageHeader)
    defi, rep, val = fastparquet.core.read_data_page(file, parquetfile.schema, ph, col.meta_data, False)
    print(defi)
    print(rep)
    print(val)
