#include "superlu_ddefs.h"
#include "adi_helper.h"

void adi(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *F, int ldf, double *F_transpose, int ldft,
    double *p, double *q, int_t l, double *X);

void adi_ls(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid, double *F, int local_b, int nrhs, double s, double la, double ua, double lb, double ub, double *X);

void adi_ls2(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *F, int ldf, double *F_transpose, int ldft, 
    int nrhs, double s, double la, double ua, double lb, double ub, double *X, int *grid_proc, int grA, int grB);

void fadi(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *U, int ldu, double *V, int ldv,
    double *p, double *q, int_t l, double tol, double **Z, double **D, double **Y, int r, int *rank);

void fadi_sp(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_B, gridinfo_t *grid_C, double *U, int ldu, double *V, int ldv,
    double *p, double *q, int_t l, double tol, double **Z, double **Y, int r, int *rank,
    double la, double ua, double lb, double ub, int *grid_proc, int grB, int grC);

void fadi_col(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    gridinfo_t *grid_A, double *U, int ldu, double *p, double *q, int_t l, double tol, double **Z, int r, int *rank);

void fadi_col_adils(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B, gridinfo_t *grid, double *F, int local_b,
    double *p, double *q, int_t l, double la, double ua, double lb, double ub, double tol, double **Z, int r, int *rank);

void fadi_ttsvd_3d(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_A, gridinfo_t *grid_B, gridinfo_t *grid_C, double *U1, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc);

void fadi_ttsvd_3d_2grids(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, double *U1, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc);

void fadi_ttsvd(superlu_dist_options_t options, int d, int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs,
    gridinfo_t **grids, double **Us, double *V, int_t *locals, int_t *nrhss, double **ps, double **qs, int_t *ls, double tol,
    double *las, double *uas, double *lbs, double *ubs, double **TTcores, int *rs, int *grid_proc);

void fadi_dimPara_ttsvd_3d(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_A, gridinfo_t *grid_B, gridinfo_t *grid_C, double *U1, int ldu1, 
    double *U2, int ldu2, double *U2T, int ldu2t, double *V2, int ldv2,
    double *p, double *q, int_t l, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc);