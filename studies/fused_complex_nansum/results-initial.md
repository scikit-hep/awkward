# Fused native complex NaN prototype

Baseline: PR #4549 at 82615e865; candidate: fused native sum/count kernel. Three alternating fresh-process pairs (B/C, C/B, B/C), seven batches of five calls, three warmups. Inputs and exact parity checks outside timing. No profiling. NumPy matrix inputs, axis=-1, default options. NaN pattern: every 100th entry in each row. All outputs checked exactly against ordinary masked sum/mean. Local CPU measurements; not against main.

| Case | Baseline µs | Fused µs | Candidate / baseline |
| --- | ---: | ---: | ---: |
| 4x8-complex64-finite-nansum | 42.5 | 13.6 | 0.320 |
| 4x8-complex64-finite-nanmean | 49.9 | 19.5 | 0.390 |
| 4x8-complex64-nan-nansum | 74.0 | 13.4 | 0.181 |
| 4x8-complex64-nan-nanmean | 84.5 | 19.4 | 0.230 |
| 4x8-complex128-finite-nansum | 42.4 | 14.4 | 0.340 |
| 4x8-complex128-finite-nanmean | 48.6 | 19.6 | 0.403 |
| 4x8-complex128-nan-nansum | 74.3 | 14.6 | 0.197 |
| 4x8-complex128-nan-nanmean | 84.8 | 19.9 | 0.235 |
| 16384x16-complex64-finite-nansum | 111.5 | 157.3 | 1.411 |
| 16384x16-complex64-finite-nanmean | 168.8 | 213.6 | 1.265 |
| 16384x16-complex64-nan-nansum | 454.8 | 160.5 | 0.353 |
| 16384x16-complex64-nan-nanmean | 596.9 | 209.0 | 0.350 |
| 16384x16-complex128-finite-nansum | 120.5 | 156.1 | 1.295 |
| 16384x16-complex128-finite-nanmean | 158.5 | 194.9 | 1.229 |
| 16384x16-complex128-nan-nansum | 627.7 | 165.8 | 0.264 |
| 16384x16-complex128-nan-nanmean | 759.5 | 205.0 | 0.270 |
| 256x1024-complex64-finite-nansum | 172.7 | 309.3 | 1.791 |
| 256x1024-complex64-finite-nanmean | 185.3 | 306.6 | 1.654 |
| 256x1024-complex64-nan-nansum | 598.0 | 309.0 | 0.517 |
| 256x1024-complex64-nan-nanmean | 644.7 | 319.5 | 0.495 |
| 256x1024-complex128-finite-nansum | 165.7 | 310.5 | 1.874 |
| 256x1024-complex128-finite-nanmean | 172.5 | 312.3 | 1.811 |
| 256x1024-complex128-nan-nansum | 703.0 | 303.7 | 0.432 |
| 256x1024-complex128-nan-nanmean | 765.4 | 309.4 | 0.404 |
| 16x100000-complex64-finite-nansum | 936.2 | 1848.2 | 1.974 |
| 16x100000-complex64-finite-nanmean | 952.0 | 1869.4 | 1.964 |
| 16x100000-complex64-nan-nansum | 3105.8 | 1848.5 | 0.595 |
| 16x100000-complex64-nan-nanmean | 3352.4 | 1849.8 | 0.552 |
| 16x100000-complex128-finite-nansum | 886.3 | 1841.3 | 2.077 |
| 16x100000-complex128-finite-nanmean | 861.9 | 1881.0 | 2.182 |
| 16x100000-complex128-nan-nansum | 4334.7 | 1873.6 | 0.432 |
| 16x100000-complex128-nan-nanmean | 4611.8 | 1856.3 | 0.403 |
