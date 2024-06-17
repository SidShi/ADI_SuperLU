#include "superlu_ddefs.h"
#include "adi_helper.h"

void dread_matrix(FILE *fp, char *postfix, gridinfo_t *grid, int_t *m, int_t *n, 
    int_t *nnz, double **nzval, int_t **rowind, int_t **colptr);

void dread_shift(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B);

void dread_shift_onegrid(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid);

void dread_shift_twogrids(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int grA, int grB, int *grid_proc);

void dread_shift_multigrids(FILE *fp, double **p, double **q, int_t *l, gridinfo_t **grids, int *grid_proc, int d);

void dread_shift_interval(FILE *fp, double *a, double *b, double *c, double *d, gridinfo_t *grid);

void dread_shift_interval_twogrids(FILE *fp, double *a, double *b, double *c, double *d, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int grA, int grB, int *grid_proc);

void dread_shift_multi_interval_2grids(FILE *fp, int d, double *la, double *ua, double *lb, double *ub, 
    gridinfo_t *grid1, gridinfo_t *grid2, int *grid_proc);

void dread_shift_interval_multigrids(FILE *fp, int d, double *la, double *ua, double *lb, double *ub, gridinfo_t **grids, int *grid_proc);

void dread_size(FILE *fp, int d, int *ms, gridinfo_t **grids);

void dread_RHS(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t *m_A, int_t *m_B, int *ldf, int *ldft, double **F_global);

void dread_RHS_multiple(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t *m_A, int_t *m_B, int_t *r, int *ldf, int *ldft, int *grid_proc, int send_grid, int recv_grid);

void dread_RHS_onegrid(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid, 
    int_t *m_A, int_t *m_B, int *ldf, int *ldft, double **F_global);

void dread_RHS_factor(FILE *fp, double **F, gridinfo_t *grid, int_t *m_global, int_t *r, int *ldf, double **F_global);

void dread_RHS_factor_twodim(FILE *fp, double **F, gridinfo_t *grid, int_t m_A, int *ldf, double **F_global);

void dread_RHS_factor_multidim(FILE *fp, double **F, gridinfo_t *grid, int_t *ms, int ddeal, int *local, double **F_global);

void dread_X(FILE *fp, double **X, gridinfo_t *grid);