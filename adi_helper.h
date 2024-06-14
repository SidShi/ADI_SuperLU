#include "superlu_ddefs.h"

void adi_grid_barrier(gridinfo_t **grids, int d);

void dtransfer_size(int_t *m_A, int_t *m_B, gridinfo_t *grid_A, gridinfo_t *grid_B);

void dcreate_SuperMatrix(SuperMatrix *A, gridinfo_t *grid, int_t m, int_t n, 
    int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s);

void dcreate_RHS(double* F, int ldb, double **rhs, int AtoB, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s,
    double* x, int ldx, int colx);

void dcreate_RHS_multiple(double* F, int ldb, int r, double **rhs, int AtoB, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s,
    double* x, int ldx, int colx, int *grid_proc, int grA, int grB);

void dcreate_RHS_ls(double* F, int local_b, int nrhs, double **rhs, gridinfo_t *grid,
    double *A, int m_A, double s, double *X);

void dcreate_RHS_ls_sp(double* F, int local_b, int nrhs, double **rhs, gridinfo_t *grid, int_t m_A,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s, double *X);

void dgather_X(double *localX, int local_ldx, double *X, int ldx, int nrhs, gridinfo_t *grid);

void dgather_X_reg(double *localX, int nrow, int local_col, double *X, gridinfo_t *grid);

void transfer_X(double *X, int ldx, int nrhs, double *rX, gridinfo_t *grid_A, int AtoB);

void transfer_X_dgrids(double *X, int ldx, int nrhs, double *rX, int *grid_proc, int send_grid, int recv_grid);

void dcheck_error(int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *RHS, double *X, double *trueX);

void dQR_dist(double *localX, int local_ldx, double **localQ, double **R, int nrhs, gridinfo_t *grid);

void dCPQR_dist_getQ(double *localX, int local_ldx, double **localQ, int nrhs, int *rank, gridinfo_t *grid, double tol);

void drecompression_dist(double *localX, int local_ldx, int global_ldx, double **localU, double *D, double **S, gridinfo_t *grid_A,
    double *localY, int local_ldy, int global_ldy, double **localV, gridinfo_t *grid_B, int *nrhs, double tol);

void drecompression_dist_twogrids(double *localX, int local_ldx, int global_ldx, double **localU, double *D, double **S, gridinfo_t *grid_A,
    double *localY, int local_ldy, int global_ldy, double **localV, gridinfo_t *grid_B, int *nrhs, double tol, int *grid_proc, int grA, int grB);

void dtruncated_SVD(double *X, int m, int n, int *r, double **U, double **S, double **V, double tol);

void dcheck_error_fadi(int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    double *F, int r, double *Z, double *D, double *Y, int rank, double *trueX);

void ellipj(double *u, double *dn, double m, int_t l);

double ellipke(double k);

void getshifts_adi(double a, double b, double c, double d, double **p, double **q, int_t *l, double tol);

void dgenerate_shifts(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B);

void dgenerate_shifts_onegrid(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid);

void dgenerate_shifts_twogrids(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int *grid_proc, int grA, int grB);

void dgather_TTcores(double **TTcores, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, double **TTcores_global);

void dgather_TTcores_2grids(double **TTcores, gridinfo_t *grid1, gridinfo_t *grid2, int *ms, int *rs, int *locals, int d, double **TTcores_global);

void dconvertTT_tensor(double **TTcores_global, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, double *X, int *grid_proc);

void dconvertTT_tensor_2grids(double **TTcores_global, gridinfo_t *grid1, gridinfo_t *grid2, int *ms, int *rs, int *locals, 
    int d, double *X, int *grid_proc);

void dcheck_error_TT(int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs, gridinfo_t **grids,
    int *rs, int *locals, int d, double *F, double **TTcores, double *trueX, int *grid_proc);

void dcheck_error_TT_2grids(int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs, gridinfo_t *grid1, gridinfo_t *grid2,
    int *rs, int *locals, int d, double *F, double **TTcores, double *trueX, int *grid_proc);

void dredistribute_X_twogrids(double *X, double *F, double *F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t m_A, int_t m_B, int_t r, int *grid_proc, int send_grid, int recv_grid);

void dTT_right_orthonormalization(double **TTcores, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, int *grid_proc);

void dTT_rounding(double **TTcores, double **TTcores_new, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, int *grid_proc, double tol);

void dmult_TTfADI_mat(int_t m_A, double *A, int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    double *U, int r, double *X);

void dmult_TTfADI_RHS(int_t *ms, int_t *rs, int_t local, int ddeal, double *M, int nrhs, double **TTcores_global, double **newM);