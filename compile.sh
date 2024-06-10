#!/bin/bash

cc -o adi_mat driver.c adi.c adi_grid.c -I/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/include -L/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/lib -lsuperlu_dist
cc -o fadi_mat driver_fadi.c adi.c adi_grid.c -I/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/include -L/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/lib -lsuperlu_dist