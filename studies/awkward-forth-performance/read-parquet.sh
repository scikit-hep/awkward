for compress in zlib1; do   # zlib0 
  for jagged in 0 1 2 3; do
    for splt in nosplit; do   #  split
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
    done
  done
done
