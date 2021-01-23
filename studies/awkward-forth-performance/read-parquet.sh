for compress in zlib0 zlib1; do
  for jagged in 0 1 2 3; do
    for splt in nosplit split; do
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python forth-read-jaggedN-parquet.py $compress $jagged $splt;
    done
  done
done
