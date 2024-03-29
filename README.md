# mir-blas
ndslice wrapper for BLAS

## Required libraries

See [wiki: Link with CBLAS & LAPACK](https://github.com/libmir/mir-lapack/wiki/Link-with-CBLAS-&-LAPACK).

## Wrapped API

### Level 1
 - dot
 - nrm2
 - asum
 - copy
 - axpy
 - scal
 - swap
 - iamax

### Level 2
 - gemv
 - ger (includes geru)
 - gerc (includes ger)
 - symv
 - spmv
 - syr
 - spr
 - trmv
 - tpmv
 - trsv
 - tpsv

### Level 3
 - gemm
 - symm
 - syrk
 - trmm
 - trsm

---------------

This work has been sponsored by [Symmetry Investments](http://symmetryinvestments.com) and [Kaleidic Associates](https://github.com/kaleidicassociates).
