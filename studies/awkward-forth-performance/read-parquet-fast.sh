for compress in zlib9; do
  for jagged in 0; do
    for splt in split nosplit; do
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
      python pyarrow-read-jaggedN-parquet.py $compress $jagged $splt;
    done
  done
done
