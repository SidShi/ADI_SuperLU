#include "superlu_ddefs.h"

void adi_gridinit_matrix(MPI_Comm Bcomm,
        int nprow_A, int npcol_A, gridinfo_t *grid_A,
        int nprow_B, int npcol_B, gridinfo_t *grid_B);

void adi_gridmap_matrix(MPI_Comm Bcomm,
        int nprow_A, int npcol_A, int usermap_A[], int ldumap_A, gridinfo_t *grid_A,
        int nprow_B, int npcol_B, int usermap_B[], int ldumap_B, gridinfo_t *grid_B);

void adi_gridexit_matrix(gridinfo_t *grid_A, gridinfo_t *grid_B);

void adi_gridinit_tensor(MPI_Comm Bcomm, int d,
        int *nprow, int *npcol, int *Np, gridinfo_t **grid);

void adi_gridmap_tensor(MPI_Comm Bcomm, int d,
        int *nprow, int *npcol, int *Np, int **usermap, int *ldumap, gridinfo_t **grid);

void adi_gridexit_tensor(gridinfo_t **grid, int d);