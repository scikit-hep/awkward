import time
#from fastavro import reader
import awkward._v2 as ak



#start_time = time.time()
#with open('avro_samples/jagged3.avro', 'rb') as fo:
#    avro_reader = reader(fo)
#    for record in avro_reader:
#    	pass
#print("With fastavro: ", time.time()-start_time)

start_time = time.time()
ak.from_avro_file(file="../avro_samples/jagged3.avro", reader_lang="ft")
print("With generated Forth: ", time.time()-start_time)
