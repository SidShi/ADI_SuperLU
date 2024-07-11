#include <math.h>
#include "superlu_ddefs.h"
#include "adi_helper.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PDDRIVE.
 *
 * This example illustrates how to use PDGSSVX with the full
 * (default) options to solve a linear system.
 *
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pdgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pddrive -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */

#define MAX(x,y) (x>y?x:y)
#define MAX3(x,y,z) MAX(MAX(x,y),z)

void adi_grid_barrier(gridinfo_t **grids, int d)
{
    int_t j;
    for (j = 0; j < d; ++j) {
        if (grids[j]->iam != -1) {
            MPI_Barrier(grids[j]->comm);
        }
    }
}

void dtransfer_size(int_t *m_A, int_t *m_B, gridinfo_t *grid_A, gridinfo_t *grid_B)
{
    int nproc_A = grid_A->nprow * grid_A->npcol;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (!global_rank) {
        MPI_Send(m_A, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == nproc_A) {
        MPI_Recv(m_A, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid_B->iam != -1) {
        MPI_Bcast(m_A, 1, MPI_INT, 0, grid_B->comm);
    }

    if (global_rank == nproc_A) {
        MPI_Send(m_B, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (!global_rank) {
        MPI_Recv(m_B, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid_A->iam != -1) {
        MPI_Bcast(m_B, 1, MPI_INT, 0, grid_A->comm);
    }
}

void dcreate_SuperMatrix(SuperMatrix *A, gridinfo_t *grid, int_t m, int_t n, 
    int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s)
{
    double   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;   /* local */
    int_t    m_loc, fst_row, nnz_loc;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
               when mod(m, p) is not zero. */ 
    int_t    row, col, i, j, relpos;
    int      iam = grid->iam;
    int_t    *marker;

    /* Compute the number of rows to be distributed to local process */
    m_loc = m / (grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * grid->nprow * grid->npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid->nprow * grid->npcol - 1)) /* last proc. gets all*/
      m_loc = m - m_loc * (grid->nprow * grid->npcol - 1);
    }

    rowptr = (int_t *) intMalloc_dist(m_loc+1);
    marker = (int_t *) intCalloc_dist(n);

    /* Get counts of each row of GA */
    for (i = 0; i < n; ++i)
      for (j = colptr[i]; j < colptr[i+1]; ++j) ++marker[rowind[j]];
    /* Set up row pointers */
    rowptr[0] = 0;
    fst_row = iam * m_loc_fst;
    nnz_loc = 0;
    for (j = 0; j < m_loc; ++j) {
      row = fst_row + j;
      rowptr[j+1] = rowptr[j] + marker[row];
      marker[j] = rowptr[j];
    }
    nnz_loc = rowptr[m_loc];

    nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
    colind = (int_t *) intMalloc_dist(nnz_loc);

    /* Transfer the matrix into the compressed row storage */
    for (i = 0; i < n; ++i) {
      for (j = colptr[i]; j < colptr[i+1]; ++j) {
    row = rowind[j];

    if ( (row>=fst_row) && (row<fst_row+m_loc) ) {
      row = row - fst_row;
      relpos = marker[row];
      colind[relpos] = i;
      if (i == rowind[j]) {
        nzval_loc[relpos] = nzval[j]-s;
      }
      else {
        nzval_loc[relpos] = nzval[j];
      }
      ++marker[row];
    }
      }
    }

    /* Set up the local A in NR_loc format */
    dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                   nzval_loc, colind, rowptr,
                   SLU_NR_loc, SLU_D, SLU_GE);
    
    SUPERLU_FREE(marker);
}

void dcreate_RHS(double* F, int ldb, double **rhs, int AtoB, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s,
    double* x, int ldx, int colx)
{
    SuperMatrix GA;              /* global A */
    double   *b_global, *recvb_global;  /* replicated on all processes */
    double   *nzval_new;
    int_t    *rowind_new, *colptr_new;   /* global */
    int_t    i, j;
    int      iam;
    char     transpose[1];
    *transpose = 'N';
    int_t    m_loc, m_loc_fst, row, fst_row;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if ((grid_A->iam == 0) || (grid_B->iam == 0)) {
        if ( !(b_global = doubleMalloc_dist(m*colx)) )
            ABORT("Malloc fails for b_global[]");
    }
    if ( !(recvb_global = doubleMalloc_dist(m*colx)) )
        ABORT("Malloc fails for recvb_global[]");

    if (AtoB) {
        if (grid_A->iam == 0) {
            dallocateA_dist(n, nnz, &nzval_new, &rowind_new, &colptr_new);
            for (i = 0; i < n; ++i) {
                for (j = colptr[i]; j < colptr[i+1]; ++j) {
                    nzval_new[j] = nzval[j] - (i == rowind[j] ? s : 0.0);
                    rowind_new[j] = rowind[j];
                }
                colptr_new[i] = colptr[i];
            }
            colptr_new[n] = colptr[n];
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval_new, rowind_new, colptr_new,
                SLU_NC, SLU_D, SLU_GE);

            sp_dgemm_dist(transpose, colx, 1.0, &GA, x, ldx, 0.0, b_global, m);

            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);
        }
    }
    else {
        if (grid_B->iam == 0) {
            dallocateA_dist(n, nnz, &nzval_new, &rowind_new, &colptr_new);
            for (i = 0; i < n; ++i) {
                for (j = colptr[i]; j < colptr[i+1]; ++j) {
                    nzval_new[j] = nzval[j] - (i == rowind[j] ? s : 0.0);
                    rowind_new[j] = rowind[j];
                }
                colptr_new[i] = colptr[i];
            }
            colptr_new[n] = colptr[n];
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval_new, rowind_new, colptr_new,
                SLU_NC, SLU_D, SLU_GE);

            sp_dgemm_dist(transpose, colx, 1.0, &GA, x, ldx, 0.0, b_global, m);

            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);
        }
    }

    transfer_X(b_global, m, colx, recvb_global, grid_A, AtoB);

    if (AtoB) {
        iam = grid_B->iam;
        if (iam != -1) {
            MPI_Bcast(recvb_global, m*colx, MPI_DOUBLE, 0, grid_B->comm);

            m_loc = colx / (grid_B->nprow * grid_B->npcol); 
            m_loc_fst = m_loc;
            fst_row = iam * m_loc_fst;
            /* When nrhs / procs is not an integer */
            if ((m_loc * grid_B->nprow * grid_B->npcol) != colx) {
                /*m_loc = m_loc+1;
                  m_loc_fst = m_loc;*/
              if (iam == (grid_B->nprow * grid_B->npcol - 1)) /* last proc. gets all*/
                  m_loc = colx - m_loc * (grid_B->nprow * grid_B->npcol - 1);
            }

            // printf("Size of m_loc is %d and ldb is %d for grid_B id %d.\n", m_loc, ldb, iam);
            // fflush(stdout);

            if ( !((*rhs) = doubleMalloc_dist(m_loc*m)) )
                ABORT("Malloc fails for rhs[]");
            for (j = 0; j < m; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;

                    // printf("Index of rhs is %d, index of F is %d with element %f, and b is %d with element %f, for grid_B id %d.\n", 
                    //     j*m_loc+i, j*ldb+i, F[j*ldb+i], row*m+j, recvb_global[row*m+j], iam);
                    // printf("Index of rhs is %d, index of F is %d, and b is %d with element %f, for grid_B id %d.\n", 
                    //     j*m_loc+i, j*ldb+i, row*m+j, recvb_global[row*m+j], iam);
                    // fflush(stdout);

                    (*rhs)[j*m_loc+i] = - F[j*ldb+i] + recvb_global[row*m+j];
                }
            }
        }
    }
    else {
        iam = grid_A->iam;
        if (iam != -1) {
            MPI_Bcast(recvb_global, m*colx, MPI_DOUBLE, 0, grid_A->comm);
        
            m_loc = colx / (grid_A->nprow * grid_A->npcol); 
            m_loc_fst = m_loc;
            fst_row = iam * m_loc_fst;
            /* When nrhs / procs is not an integer */
            if ((m_loc * grid_A->nprow * grid_A->npcol) != colx) {
                /*m_loc = m_loc+1;
                  m_loc_fst = m_loc;*/
              if (iam == (grid_A->nprow * grid_A->npcol - 1)) /* last proc. gets all*/
                  m_loc = colx - m_loc * (grid_A->nprow * grid_A->npcol - 1);
            }

            if ( !((*rhs) = doubleMalloc_dist(m_loc*m)) )
                ABORT("Malloc fails for rhs[]");
            for (j = 0; j < m; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;
                    (*rhs)[j*m_loc+i] = F[j*ldb+i] + recvb_global[row*m+j];
                }
            }
        }
    }

    if ((grid_A->iam == 0) || (grid_B->iam == 0)) {
        SUPERLU_FREE(b_global);
    }
    SUPERLU_FREE(recvb_global);
}

void dcreate_RHS_multiple(double* F, int ldb, int r, double **rhs, int AtoB, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s,
    double* x, int ldx, int colx, int *grid_proc, int grA, int grB)
{
    SuperMatrix GA;              /* global A */
    double   *b_global, *recvb_global;  /* replicated on all processes */
    double   *nzval_new;
    int_t    *rowind_new, *colptr_new;   /* global */
    int_t    i, j, k;
    int      iam;
    char     transpose[1];
    *transpose = 'N';
    int_t    m_loc, m_loc_fst, row, fst_row;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if ((grid_A->iam == 0) || (grid_B->iam == 0)) {
        if ( !(b_global = doubleMalloc_dist(m*colx*r)) )
            ABORT("Malloc fails for b_global[]");
    }
    if ( !(recvb_global = doubleMalloc_dist(m*colx*r)) )
        ABORT("Malloc fails for recvb_global[]");

    if (AtoB) {
        if (grid_A->iam == 0) {
            dallocateA_dist(n, nnz, &nzval_new, &rowind_new, &colptr_new);
            for (i = 0; i < n; ++i) {
                for (j = colptr[i]; j < colptr[i+1]; ++j) {
                    nzval_new[j] = nzval[j] - (i == rowind[j] ? s : 0.0);
                    rowind_new[j] = rowind[j];
                }
                colptr_new[i] = colptr[i];
            }
            colptr_new[n] = colptr[n];
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval_new, rowind_new, colptr_new,
                SLU_NC, SLU_D, SLU_GE);

            sp_dgemm_dist(transpose, colx*r, 1.0, &GA, x, ldx, 0.0, b_global, m);

            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);
        }
        transfer_X_dgrids(b_global, m, colx*r, recvb_global, grid_proc, grA, grB);
    }
    else {
        if (grid_B->iam == 0) {
            dallocateA_dist(n, nnz, &nzval_new, &rowind_new, &colptr_new);
            for (i = 0; i < n; ++i) {
                for (j = colptr[i]; j < colptr[i+1]; ++j) {
                    nzval_new[j] = nzval[j] - (i == rowind[j] ? s : 0.0);
                    rowind_new[j] = rowind[j];
                }
                colptr_new[i] = colptr[i];
            }
            colptr_new[n] = colptr[n];
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval_new, rowind_new, colptr_new,
                SLU_NC, SLU_D, SLU_GE);

            sp_dgemm_dist(transpose, colx*r, 1.0, &GA, x, ldx, 0.0, b_global, m);

            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);
        }
        transfer_X_dgrids(b_global, m, colx*r, recvb_global, grid_proc, grB, grA);
    }

    if (AtoB) {
        iam = grid_B->iam;
        if (iam != -1) {
            MPI_Bcast(recvb_global, m*colx*r, MPI_DOUBLE, 0, grid_B->comm);

            m_loc = colx / (grid_B->nprow * grid_B->npcol); 
            m_loc_fst = m_loc;
            fst_row = iam * m_loc_fst;
            /* When nrhs / procs is not an integer */
            if ((m_loc * grid_B->nprow * grid_B->npcol) != colx) {
                /*m_loc = m_loc+1;
                  m_loc_fst = m_loc;*/
              if (iam == (grid_B->nprow * grid_B->npcol - 1)) /* last proc. gets all*/
                  m_loc = colx - m_loc * (grid_B->nprow * grid_B->npcol - 1);
            }

            // printf("Size of m_loc is %d and ldb is %d for grid_B id %d.\n", m_loc, ldb, iam);
            // fflush(stdout);

            if ( !((*rhs) = doubleMalloc_dist(m_loc*m*r)) )
                ABORT("Malloc fails for rhs[]");
            for (k = 0; k < r; ++k) {
                for (j = 0; j < m; ++j) {
                    for (i = 0; i < m_loc; ++i) {
                        row = fst_row + i;
                        (*rhs)[k*m*m_loc+j*m_loc+i] = - F[k*ldb*colx+j*ldb+i] + recvb_global[k*m*colx+row*m+j];
                    }
                }
            }
        }
    }
    else {
        iam = grid_A->iam;
        if (iam != -1) {
            MPI_Bcast(recvb_global, m*colx*r, MPI_DOUBLE, 0, grid_A->comm);
        
            m_loc = colx / (grid_A->nprow * grid_A->npcol); 
            m_loc_fst = m_loc;
            fst_row = iam * m_loc_fst;
            /* When nrhs / procs is not an integer */
            if ((m_loc * grid_A->nprow * grid_A->npcol) != colx) {
                /*m_loc = m_loc+1;
                  m_loc_fst = m_loc;*/
              if (iam == (grid_A->nprow * grid_A->npcol - 1)) /* last proc. gets all*/
                  m_loc = colx - m_loc * (grid_A->nprow * grid_A->npcol - 1);
            }

            if ( !((*rhs) = doubleMalloc_dist(m_loc*m*r)) )
                ABORT("Malloc fails for rhs[]");
            for (k = 0; k < r; ++k) {
                for (j = 0; j < m; ++j) {
                    for (i = 0; i < m_loc; ++i) {
                        row = fst_row + i;
                        (*rhs)[k*m_loc*m+j*m_loc+i] = F[k*ldb*colx+j*ldb+i] + recvb_global[k*m*colx+row*m+j];
                    }
                }
            }
        }
    }

    if ((grid_A->iam == 0) || (grid_B->iam == 0)) {
        SUPERLU_FREE(b_global);
    }
    SUPERLU_FREE(recvb_global);
}

void dcreate_RHS_ls(double* F, int local_b, int nrhs, double **rhs, gridinfo_t *grid,
    double *A, int m_A, double s, double *X)
{
    int_t    i, j, k;
    int      iam = grid->iam;
    double   *newA, *combined_product;
    double   one = 1.0, zero = 0.0;
    int_t    mX = local_b*nrhs;

    if ( !((*rhs) = doubleMalloc_dist(local_b*m_A*nrhs)) )
        ABORT("Malloc fails for rhs[]");

    if ( !(newA = doubleMalloc_dist(m_A*m_A)) )
        ABORT("Malloc fails for newA[]");
    for (j = 0; j < m_A; ++j) {
        for (i = 0; i < m_A; ++i) {
            newA[j*m_A+i] = A[j*m_A+i] - (i == j ? s : 0);
        }
    }
    if ( !(combined_product = doubleMalloc_dist(local_b*nrhs*m_A) ))
        ABORT("Malloc fails for combined_product[]");
    dgemm_("N", "T", &mX, &m_A, &m_A, &one, X, &mX, newA, &m_A, &zero, combined_product, &mX);

    for (k = 0; k < nrhs; ++k) {
        for (j = 0; j < m_A; ++j) {
            for (i = 0; i < local_b; ++i) {
                (*rhs)[k*local_b*m_A+j*local_b+i] = combined_product[j*mX+k*local_b+i] - F[k*local_b*m_A+i*m_A+j];
            }
        }
    }

    SUPERLU_FREE(newA);
    SUPERLU_FREE(combined_product);
}

void dcreate_RHS_ls_sp(double* F, int local_b, int nrhs, double **rhs, gridinfo_t *grid, int_t m_A,
    int_t m, int_t n, int_t nnz, double *nzval, int_t *rowind, int_t *colptr, double s, double *X)
{
    SuperMatrix GA;              /* global A */
    double   *b_global;  /* replicated on all processes */
    double   *nzval_new;
    int_t    *rowind_new, *colptr_new;   /* global */
    int_t    i, j, k;
    int      iam = grid->iam;
    char     transpose[1];
    *transpose = 'N';
    int_t    m_loc, m_loc_fst, row, fst_row;

    if ( !(b_global = doubleMalloc_dist(m*m_A*nrhs)) )
        ABORT("Malloc fails for b_global[]");

    if (!iam) {  
        dallocateA_dist(n, nnz, &nzval_new, &rowind_new, &colptr_new);
        for (i = 0; i < n; ++i) {
            for (j = colptr[i]; j < colptr[i+1]; ++j) {
                nzval_new[j] = nzval[j] - (i == rowind[j] ? s : 0.0);
                rowind_new[j] = rowind[j];
            }
            colptr_new[i] = colptr[i];
        }
        colptr_new[n] = colptr[n];
        /* Create compressed column matrix for GA. */
        dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval_new, rowind_new, colptr_new,
            SLU_NC, SLU_D, SLU_GE);

        sp_dgemm_dist(transpose, m_A*nrhs, 1.0, &GA, X, m, 0.0, b_global, m);

        /* Destroy GA */
        Destroy_CompCol_Matrix_dist(&GA);
    }
    MPI_Bcast(b_global, m*m_A*nrhs, MPI_DOUBLE, 0, grid->comm);

    m_loc = m / (grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    fst_row = iam * m_loc_fst;
    /* When nrhs / procs is not an integer */
    if ((m_loc * grid->nprow * grid->npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid->nprow * grid->npcol - 1)) /* last proc. gets all*/
          m_loc = m - m_loc * (grid->nprow * grid->npcol - 1);
    }

    // printf("Size of m_loc is %d and ldb is %d for grid_B id %d.\n", m_loc, ldb, iam);
    // fflush(stdout);

    if ( !((*rhs) = doubleMalloc_dist(m_A*m_loc*nrhs)) )
        ABORT("Malloc fails for *rhs[]");
    for (k = 0; k < nrhs; ++k) {
        for (j = 0; j < m_loc; ++j) {
            for (i = 0; i < m_A; ++i) {
                row = fst_row + j;
                (*rhs)[k*m_loc*m_A+j*m_A+i] = F[k*m_loc*m_A+j*m_A+i] + b_global[k*m*m_A+i*m+row];
            }
        }
    }

    SUPERLU_FREE(b_global);
}

void dgather_X(double *localX, int local_ldx, double *X, int ldx, int nrhs, gridinfo_t *grid)
{
    int_t    i, j, ldx_disp, p;
    int      iam = grid->iam;
    int      nproc = grid->nprow * grid->npcol;
    double   *Y;
    int      *local_ldx_count, *local_X_disp, *local_X_count;

    if ( !(local_ldx_count = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for ldx_count[]");

    MPI_Gather(&local_ldx, 1, MPI_INT, local_ldx_count, 1, MPI_INT, 0, grid->comm);

    if ( !(local_X_disp = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for X_disp[]");
    if ( !(local_X_count = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for local_X[]");
    if (iam == 0) {
        local_X_count[0] = local_ldx_count[0]*nrhs;
        local_X_disp[0] = 0;
        for (i = 0; i < nproc-1; ++i) {
            local_X_count[i+1] = local_ldx_count[i+1]*nrhs;
            local_X_disp[i+1] = local_X_disp[i] + local_X_count[i];
        }
    }

    if ( !(Y = doubleMalloc_dist(ldx*nrhs)) )
        ABORT("Malloc fails for Y[]");
    MPI_Gatherv(localX, local_ldx*nrhs, MPI_DOUBLE, Y, local_X_count, local_X_disp, MPI_DOUBLE, 0, grid->comm);

    if (iam == 0) {
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < nrhs; ++j) {
                for (i = 0; i < local_ldx_count[p]; ++i) {
                    ldx_disp = local_X_disp[p] / nrhs;
                    X[j*ldx+i+ldx_disp] = Y[local_X_disp[p]+j*local_ldx_count[p]+i];
                }
            }
        }
    }

    SUPERLU_FREE(local_ldx_count);
    SUPERLU_FREE(local_X_count);
    SUPERLU_FREE(local_X_disp);
    SUPERLU_FREE(Y);
}

void dgather_X_reg(double *localX, int nrow, int local_col, double *X, gridinfo_t *grid)
{
    int_t    i, j, ldx_disp, p;
    int      iam = grid->iam;
    int      nproc = grid->nprow * grid->npcol;
    int      *local_ldx_count, *local_X_disp, *local_X_count;

    if ( !(local_ldx_count = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for ldx_count[]");

    MPI_Gather(&local_col, 1, MPI_INT, local_ldx_count, 1, MPI_INT, 0, grid->comm);

    if ( !(local_X_disp = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for X_disp[]");
    if ( !(local_X_count = intMalloc_dist(nproc)) )
        ABORT("Malloc fails for local_X[]");
    if (iam == 0) {
        local_X_count[0] = local_ldx_count[0]*nrow;
        local_X_disp[0] = 0;
        for (i = 0; i < nproc-1; ++i) {
            local_X_count[i+1] = local_ldx_count[i+1]*nrow;
            local_X_disp[i+1] = local_X_disp[i] + local_X_count[i];
        }
    }

    MPI_Gatherv(localX, nrow*local_col, MPI_DOUBLE, X, local_X_count, local_X_disp, MPI_DOUBLE, 0, grid->comm);

    SUPERLU_FREE(local_ldx_count);
    SUPERLU_FREE(local_X_count);
    SUPERLU_FREE(local_X_disp);
}

void transfer_X(double *X, int ldx, int nrhs, double *rX, gridinfo_t *grid_A, int AtoB)
{
    int nproc_A = grid_A->nprow * grid_A->npcol;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (AtoB) {
        if (!global_rank) {
            MPI_Send(X, ldx*nrhs, MPI_DOUBLE, nproc_A, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == nproc_A) {
            MPI_Recv(rX, ldx*nrhs, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        if (!global_rank) {
            MPI_Recv(rX, ldx*nrhs, MPI_DOUBLE, nproc_A, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (global_rank == nproc_A) {
            MPI_Send(X, ldx*nrhs, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void transfer_X_dgrids(double *X, int ldx, int nrhs, double *rX, int *grid_proc, int send_grid, int recv_grid)
{
    int_t j;
    int global_rank, send_rank = 0, recv_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    for (j = 0; j < send_grid; ++j) {
        send_rank += grid_proc[j];
    }
    for (j = 0; j < recv_grid; ++j) {
        recv_rank += grid_proc[j];
    }

    if (global_rank == send_rank) {
        MPI_Send(X, ldx*nrhs, MPI_DOUBLE, recv_rank, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == recv_rank) {
        MPI_Recv(rX, ldx*nrhs, MPI_DOUBLE, send_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void dcheck_error(int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *RHS, double *X, double *trueX)
{
    SuperMatrix GA, GB;
    char     transpose[1];
    *transpose = 'N';
    double *X_transpose, *rX, *AX, *BXT, *rBXT;
    int iam_A = grid_A->iam;
    int iam_B = grid_B->iam;
    int_t i, j;
    double err1 = 0.0, err2 = 0.0, norm1 = 0.0, norm2 = 0.0;
    double s;

    // if (iam_A == 0) {
    //     printf("Solution is\n");
    //     for (i = 0; i < m_A; ++i) {
    //         for (j = 0; j < m_B; ++j) {
    //             printf("%f ", X[j*m_A+i]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }

    if (iam_A == 0) {
        /* Create compressed column matrix for GA. */
        dCreate_CompCol_Matrix_dist(&GA, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A,
            SLU_NC, SLU_D, SLU_GE);

        if ( !(AX = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for AX[]");
        sp_dgemm_dist(transpose, m_B, 1.0, &GA, X, m_A, 0.0, AX, m_A);

        /* Destroy GA */
        Destroy_CompCol_Matrix_dist(&GA);
    }

    if (iam_B == 0) {
        if ( !(rX = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for rX");
    }
    transfer_X(X, m_A, m_B, rX, grid_A, 1);

    if (iam_B == 0) {
        /* Create compressed column matrix for GA. */
        dCreate_CompCol_Matrix_dist(&GB, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
            SLU_NC, SLU_D, SLU_GE);

        if ( !(X_transpose = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for X_transpose[]");
        if ( !(BXT = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for BXT[]");


        for (j = 0; j < m_A; ++j) {
            for (i = 0; i < m_B; ++i) {
                X_transpose[j*m_B+i] = rX[i*m_A+j];
            }
        }
        sp_dgemm_dist(transpose, m_A, 1.0, &GB, X_transpose, m_B, 0.0, BXT, m_B);

        /* Destroy GA */
        Destroy_CompCol_Matrix_dist(&GB);
    }

    if (iam_A == 0) {
        if ( !(rBXT = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for rBXT[]");
        // if ( !(RHS = doubleMalloc_dist(m_A*m_B)) )
        //     ABORT("Malloc fails for RHS[]");
    }
    transfer_X(BXT, m_B, m_A, rBXT, grid_A, 0);
    // if (iam_A != -1) {
    //     dgather_X(F, ldf, RHS, m_A, m_B, grid_A);
    // }

    if (iam_A == 0) {
        for (j = 0; j < m_B; ++j) {
            for (i = 0; i < m_A; ++i) {
                s = RHS[j*m_A+i] - (AX[j*m_A+i] - rBXT[i*m_B+j]);
                err1 += s*s;

                s = X[j*m_A+i] - trueX[j*m_A+i];
                err2 += s*s;

                s = RHS[j*m_A+i];
                norm1 += s*s;

                s = trueX[j*m_A+i];
                norm2 += s*s;
            }
        }
        err1 = sqrt(err1) / sqrt(norm1);
        err2 = sqrt(err2) / sqrt(norm2);
        printf("Relative error of approximating RHS is %f, and approximating true solution is %f\n.", err1, err2);
        fflush(stdout);

        // SUPERLU_FREE(RHS);
        SUPERLU_FREE(rBXT);
        SUPERLU_FREE(AX);

        // printf("Trying QR\n.");
        // fflush(stdout);

        // double *tau, *WORK;
        // int len = m_A <= m_B ? m_A : m_B;
        // if ( !(tau = doubleMalloc_dist(len)) )
        //     ABORT("Malloc fails for tau[]");
        // int INFO;
        // int LWORK = m_B;
        // if ( !(WORK = doubleMalloc_dist(LWORK)) )
        //     ABORT("Malloc fails for WORK[]");
        // dgeqrf_(&m_A, &m_B, trueX, &m_A, tau, WORK, &LWORK, &INFO);

        // printf("Tried QR\n.");
        // fflush(stdout);
    }

    if (iam_B == 0) {
        SUPERLU_FREE(BXT);
        SUPERLU_FREE(X_transpose);
        SUPERLU_FREE(rX);
    }
}

void dQR_dist(double *localX, int local_ldx, double **localQ, double **R, int nrhs, gridinfo_t *grid)
{
    int_t    i, j, k, p, ct, offset, rnum, rnum_total;
    int      iam = grid->iam;
    int      nproc = grid->nprow * grid->npcol;
    int      rnum_comb[nproc], localR_disp[nproc];
    int      m_local_comb[nproc], m_local_disp[nproc], m_local_scatter[nproc], m_local_scatter_disp[nproc];
    double   *localQ_1, *temp_localQ, *localR, *gathered_localR, *gathered_localR_mat, *gathered_globalQ;
    double   *tau_local, *WORK_local, *tau_global, *WORK_global;
    int      INFO_local, LWORK_local, INFO_global, LWORK_global;
    int      m_local = local_ldx, n_local = nrhs, n_global = nrhs;
    int      lengtau_local = m_local <= n_local ? m_local : n_local;
    int      m_global;
    double   one = 1.0, zero = 0.0, WKOPT_local, WKOPT_global;

    if ( !(temp_localQ = doubleMalloc_dist(local_ldx*nrhs)) )
        ABORT("Malloc fails for temp_localQ[]");
    for (j = 0; j < nrhs; ++j) {
        for (i = 0; i < local_ldx; ++i) {
            temp_localQ[j*local_ldx+i] = localX[j*local_ldx+i];
        }
    }

    // if (grid->iam == 0) {
    //     printf("Before local QR is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*local_ldx+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    LWORK_local = -1;
    if ( !(tau_local = doubleMalloc_dist(lengtau_local)) )
        ABORT("Malloc fails for tau_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, &WKOPT_local, &LWORK_local, &INFO_local);
    LWORK_local = (int) WKOPT_local;
    if ( !(WORK_local = doubleMalloc_dist(LWORK_local)) )
        ABORT("Malloc fails for WORK_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, WORK_local, &LWORK_local, &INFO_local);

    rnum = m_local >= n_local ? nrhs*(nrhs+1)/2 : m_local*(nrhs+nrhs-m_local+1)/2;
    if ( !(localR = doubleMalloc_dist(rnum)) )
        ABORT("Malloc fails for localR[]");
    ct = 0;
    for (j = 0; j < nrhs; ++j) {
        k = j >= lengtau_local ? lengtau_local-1 : j;
        for (i = 0; i <= k; ++i) {
            localR[ct] = temp_localQ[j*local_ldx+i];
            ct++;
        }
    }

    // if (grid->iam == 0) {
    //     printf("After local QR is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*local_ldx+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    MPI_Gather(&rnum, 1, MPI_INT, rnum_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        localR_disp[0] = 0;
        for (j = 1; j < nproc; ++j) {
            localR_disp[j] = localR_disp[j-1] + rnum_comb[j-1];
        }
        rnum_total = localR_disp[nproc-1] + rnum_comb[nproc-1];
    
        if ( !(gathered_localR = doubleMalloc_dist(rnum_total)) )
            ABORT("Malloc fails for gathered_localR[]");
    }
    MPI_Gatherv(localR, rnum, MPI_DOUBLE, gathered_localR, rnum_comb, localR_disp, MPI_DOUBLE, 0, grid->comm);
    // MPI_Gather(localR, rnum, MPI_DOUBLE, gathered_localR, rnum, MPI_DOUBLE, 0, grid->comm);
    MPI_Gather(&lengtau_local, 1, MPI_INT, m_local_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        // printf("Gathered R elements are\n");
        // for (j = 0; j < nproc*rnum; ++j) {
        //     printf("%f ", gathered_localR[j]);
        // }
        // fflush(stdout);
        m_local_disp[0] = 0;
        for (p = 1; p < nproc; ++p) {
            m_local_disp[p] = m_local_disp[p-1] + m_local_comb[p-1];
        }
        m_global = m_local_disp[nproc-1] + m_local_comb[nproc-1];

        if ( !(gathered_localR_mat = doubleMalloc_dist(m_global*nrhs)) )
            ABORT("Malloc fails for gathered_localR_mat[]");
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < nrhs; ++j) {
                offset = j < m_local_comb[p] ? j*(j+1)/2 : m_local_comb[p]*(m_local_comb[p]+1)/2+(j-m_local_comb[p])*m_local_comb[p];
                for (i = 0; i < m_local_comb[p]; ++i) {
                    gathered_localR_mat[j*m_global+m_local_disp[p]+i] = i <= j ? gathered_localR[localR_disp[p]+offset+i] : 0.0;
                }
            }
        }

        // printf("QRing matrix on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        LWORK_global = -1;
        if ( !(tau_global = doubleMalloc_dist(nrhs)) )
            ABORT("Malloc fails for tau_global[]");
        dgeqrf_(&m_global, &n_global, gathered_localR_mat, &m_global, tau_global, &WKOPT_global, &LWORK_global, &INFO_global);
        LWORK_global = (int) WKOPT_global;
        if ( !(WORK_global = doubleMalloc_dist(LWORK_global)) )
            ABORT("Malloc fails for WORK_global[]");
        dgeqrf_(&m_global, &n_global, gathered_localR_mat, &m_global, tau_global, WORK_global, &LWORK_global, &INFO_global);

        // printf("QRed matrix on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        if ( !((*R) = doubleMalloc_dist(nrhs*nrhs)) )
            ABORT("Malloc fails for *R[]");
        for (j = 0; j < nrhs; ++j) {
            for (i = 0; i < nrhs; ++i) {
                (*R)[j*nrhs+i] = i <= j ? gathered_localR_mat[j*m_global+i] : 0.0;
            }
        }

        // printf("R on the root is\n");
        // for (i = 0; i < n_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", (*R)[j*n_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        dorgqr_(&m_global, &n_global, &n_global, gathered_localR_mat, &m_global, tau_global, WORK_global, &LWORK_global, &INFO_global);

        // printf("Q on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        if ( !(gathered_globalQ = doubleMalloc_dist(m_global*nrhs)) )
            ABORT("Malloc fails for gathered_globalQ[]");
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < nrhs; ++j) {
                for (i = 0; i < m_local_comb[p]; ++i) {
                    gathered_globalQ[nrhs*m_local_disp[p]+j*m_local_comb[p]+i] = gathered_localR_mat[j*m_global+m_local_disp[p]+i];
                }
            }
            m_local_scatter[p] = m_local_comb[p] * nrhs;
            m_local_scatter_disp[p] = m_local_disp[p] * nrhs;
        }

        SUPERLU_FREE(gathered_localR);
        SUPERLU_FREE(gathered_localR_mat);
        SUPERLU_FREE(tau_global);
        SUPERLU_FREE(WORK_global);
    }
    if ( !(localQ_1 = doubleMalloc_dist(lengtau_local*nrhs)) )
        ABORT("Malloc fails for localQ_1[]");
    MPI_Scatterv(gathered_globalQ, m_local_scatter, m_local_scatter_disp, MPI_DOUBLE, localQ_1, lengtau_local*nrhs, MPI_DOUBLE, 0, grid->comm);
    // MPI_Scatter(gathered_globalQ, lengtau_local*nrhs, MPI_DOUBLE, localQ_1, lengtau_local*nrhs, MPI_DOUBLE, 0, grid->comm);

    dorgqr_(&m_local, &lengtau_local, &lengtau_local, temp_localQ, &m_local, tau_local, WORK_local, &LWORK_local, &INFO_local);

    // if (grid->iam == 0) {
    //     printf("local Q on the root is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*m_local+i]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }

    if ( !((*localQ) = doubleMalloc_dist(local_ldx*nrhs)) )
        ABORT("Malloc fails for *localQ[]");

    dgemm_("N", "N", &m_local, &n_local, &lengtau_local, &one, temp_localQ, &m_local, localQ_1, &lengtau_local, &zero, *localQ, &m_local);

    SUPERLU_FREE(temp_localQ);
    SUPERLU_FREE(tau_local);
    SUPERLU_FREE(WORK_local);
    SUPERLU_FREE(localR);
    SUPERLU_FREE(localQ_1);
    if (!iam) {
        SUPERLU_FREE(gathered_globalQ);
    }
}

void dCPQR_dist_getQ(double *localX, int local_ldx, double **localQ, int nrhs, int *rank, gridinfo_t *grid, double tol)
{
    int_t    i, j, k, p, ct, offset, rnum, rnum_total, newr;
    int      iam = grid->iam;
    int      nproc = grid->nprow * grid->npcol;
    int      rnum_comb[nproc], localR_disp[nproc];
    int      m_local_comb[nproc], m_local_disp[nproc], m_local_scatter[nproc], m_local_scatter_disp[nproc];
    double   *localQ_1, *temp_localQ, *localR, *gathered_localR, *gathered_localR_mat, *gathered_globalQ;
    int      *localPerm;
    double   *tau_local, *WORK_local, *tau_global, *WORK_global;
    int      INFO_local, LWORK_local, INFO_global, LWORK_global;
    int      m_local = local_ldx, n_local = nrhs, n_global = nrhs;
    int      lengtau_local = m_local <= n_local ? m_local : n_local;
    int      m_global, lengtau_global;
    double   one = 1.0, zero = 0.0, WKOPT_local, WKOPT_global;

    if ( !(temp_localQ = doubleMalloc_dist(local_ldx*nrhs)) )
        ABORT("Malloc fails for temp_localQ[]");
    for (j = 0; j < nrhs; ++j) {
        for (i = 0; i < local_ldx; ++i) {
            temp_localQ[j*local_ldx+i] = localX[j*local_ldx+i];
        }
    }

    // if (grid->iam == 0) {
    //     printf("Before local QR is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*local_ldx+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    LWORK_local = -1;
    if ( !(tau_local = doubleMalloc_dist(lengtau_local)) )
        ABORT("Malloc fails for tau_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, &WKOPT_local, &LWORK_local, &INFO_local);
    LWORK_local = (int) WKOPT_local;
    if ( !(WORK_local = doubleMalloc_dist(LWORK_local)) )
        ABORT("Malloc fails for WORK_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, WORK_local, &LWORK_local, &INFO_local);

    rnum = m_local >= n_local ? nrhs*(nrhs+1)/2 : m_local*(nrhs+nrhs-m_local+1)/2;
    if ( !(localR = doubleMalloc_dist(rnum)) )
        ABORT("Malloc fails for localR[]");
    ct = 0;
    for (j = 0; j < nrhs; ++j) {
        k = j >= lengtau_local ? lengtau_local-1 : j;
        for (i = 0; i <= k; ++i) {
            localR[ct] = temp_localQ[j*local_ldx+i];
            ct++;
        }
    }

    // if (grid->iam == 0) {
    //     printf("After local QR is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*local_ldx+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    MPI_Gather(&rnum, 1, MPI_INT, rnum_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        localR_disp[0] = 0;
        for (j = 1; j < nproc; ++j) {
            localR_disp[j] = localR_disp[j-1] + rnum_comb[j-1];
        }
        rnum_total = localR_disp[nproc-1] + rnum_comb[nproc-1];
    
        if ( !(gathered_localR = doubleMalloc_dist(rnum_total)) )
            ABORT("Malloc fails for gathered_localR[]");
    }
    MPI_Gatherv(localR, rnum, MPI_DOUBLE, gathered_localR, rnum_comb, localR_disp, MPI_DOUBLE, 0, grid->comm);
    // MPI_Gather(localR, rnum, MPI_DOUBLE, gathered_localR, rnum, MPI_DOUBLE, 0, grid->comm);
    MPI_Gather(&lengtau_local, 1, MPI_INT, m_local_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        if ( !(gathered_localR = doubleMalloc_dist(nproc*rnum)) )
            ABORT("Malloc fails for gathered_localR[]");
    }
    MPI_Gather(localR, rnum, MPI_DOUBLE, gathered_localR, rnum, MPI_DOUBLE, 0, grid->comm);
    if (!iam) {
        // printf("Gathered R elements are\n");
        // for (j = 0; j < nproc*rnum; ++j) {
        //     printf("%f ", gathered_localR[j]);
        // }
        // fflush(stdout);

        m_local_disp[0] = 0;
        for (p = 1; p < nproc; ++p) {
            m_local_disp[p] = m_local_disp[p-1] + m_local_comb[p-1];
        }
        m_global = m_local_disp[nproc-1] + m_local_comb[nproc-1];

        if ( !(gathered_localR_mat = doubleMalloc_dist(m_global*nrhs)) )
            ABORT("Malloc fails for gathered_localR_mat[]");
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < nrhs; ++j) {
                offset = j < m_local_comb[p] ? j*(j+1)/2 : m_local_comb[p]*(m_local_comb[p]+1)/2+(j-m_local_comb[p])*m_local_comb[p];
                for (i = 0; i < m_local_comb[p]; ++i) {
                    gathered_localR_mat[j*m_global+m_local_disp[p]+i] = i <= j ? gathered_localR[localR_disp[p]+offset+i] : 0.0;
                }
            }
        }

        // printf("QRing matrix on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        LWORK_global = -1;
        lengtau_global = m_global <= n_global ? m_global : n_global;
        if ( !(tau_global = doubleMalloc_dist(lengtau_global)) )
            ABORT("Malloc fails for tau_global[]");
        if ( !(localPerm = intMalloc_dist(nrhs)) )
            ABORT("Malloc fails for localPerm[]");
        for (j = 0; j < nrhs; ++j) {
            localPerm[j] = 0;
        }
        dgeqp3_(&m_global, &n_global, gathered_localR_mat, &m_global, localPerm, tau_global, &WKOPT_global, &LWORK_global, &INFO_global);
        LWORK_global = (int) WKOPT_global;
        if ( !(WORK_global = doubleMalloc_dist(LWORK_global)) )
            ABORT("Malloc fails for WORK_global[]");
        dgeqp3_(&m_global, &n_global, gathered_localR_mat, &m_global, localPerm, tau_global, WORK_global, &LWORK_global, &INFO_global);

        // printf("QRed matrix on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        newr = 0;
        for (j = 0; j < lengtau_global; ++j) {
            // printf("%dth diagonal element is %f.\n", j+1, gathered_localR_mat[j*m_global+j]);
            if (fabs(gathered_localR_mat[j*m_global+j]) <= tol) {
                break;
            }
            newr++;
        }
        // printf("Truncated rank is %d.\n", newr);
        // fflush(stdout);
        MPI_Bcast(&newr, 1, MPI_INT, 0, grid->comm);

        dorgqr_(&m_global, &lengtau_global, &lengtau_global, gathered_localR_mat, &m_global, tau_global, WORK_global, &LWORK_global, &INFO_global);

        // printf("Q on the root is\n");
        // for (i = 0; i < m_global; ++i) {
        //     for (j = 0; j < n_global; ++j) {
        //         printf("%f ", gathered_localR_mat[j*m_global+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        if ( !(gathered_globalQ = doubleMalloc_dist(m_global*newr)) )
            ABORT("Malloc fails for gathered_globalQ[]");
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < m_local_comb[p]; ++i) {
                    gathered_globalQ[newr*m_local_disp[p]+j*m_local_comb[p]+i] = gathered_localR_mat[j*m_global+m_local_disp[p]+i];
                }
            }
            m_local_scatter[p] = m_local_comb[p] * newr;
            m_local_scatter_disp[p] = m_local_disp[p] * newr;
        }

        SUPERLU_FREE(gathered_localR);
        SUPERLU_FREE(gathered_localR_mat);
        SUPERLU_FREE(tau_global);
        SUPERLU_FREE(WORK_global);
        SUPERLU_FREE(localPerm);
    }
    else {
        MPI_Bcast(&newr, 1, MPI_INT, 0, grid->comm);
    }

    if ( !(localQ_1 = doubleMalloc_dist(lengtau_local*newr)) )
        ABORT("Malloc fails for localQ_1[]");
    MPI_Scatter(gathered_globalQ, lengtau_local*newr, MPI_DOUBLE, localQ_1, lengtau_local*newr, MPI_DOUBLE, 0, grid->comm);

    dorgqr_(&m_local, &lengtau_local, &lengtau_local, temp_localQ, &m_local, tau_local, WORK_local, &LWORK_local, &INFO_local);

    // if (grid->iam == 0) {
    //     printf("local Q on the root is\n");
    //     for (i = 0; i < m_local; ++i) {
    //         for (j = 0; j < n_local; ++j) {
    //             printf("%f ", temp_localQ[j*m_local+i]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }

    if ( !((*localQ) = doubleMalloc_dist(local_ldx*newr)) )
        ABORT("Malloc fails for *localQ[]");
    dgemm_("N", "N", &m_local, &newr, &lengtau_local, &one, temp_localQ, &m_local, localQ_1, &lengtau_local, &zero, *localQ, &m_local);

    *rank = newr;

    SUPERLU_FREE(temp_localQ);
    SUPERLU_FREE(tau_local);
    SUPERLU_FREE(WORK_local);
    SUPERLU_FREE(localR);
    SUPERLU_FREE(localQ_1);
    if (!iam) {
        SUPERLU_FREE(gathered_globalQ);
    }
}

void dCPQR_dist_getrank(double *localX, int local_ldx, int nrhs, int *rank, gridinfo_t *grid, double tol)
{
    int_t    i, j, k, p, ct, offset, rnum, rnum_total, newr;
    int      iam = grid->iam;
    int      nproc = grid->nprow * grid->npcol;
    int      rnum_comb[nproc], localR_disp[nproc];
    int      m_local_comb[nproc], m_local_disp[nproc], m_local_scatter[nproc], m_local_scatter_disp[nproc];
    double   *temp_localQ, *localR, *gathered_localR, *gathered_localR_mat;
    int      *localPerm;
    double   *tau_local, *WORK_local, *tau_global, *WORK_global;
    int      INFO_local, LWORK_local, INFO_global, LWORK_global;
    int      m_local = local_ldx, n_local = nrhs, n_global = nrhs;
    int      lengtau_local = m_local <= n_local ? m_local : n_local;
    int      m_global, lengtau_global;
    double   one = 1.0, zero = 0.0, WKOPT_local, WKOPT_global;

    if ( !(temp_localQ = doubleMalloc_dist(local_ldx*nrhs)) )
        ABORT("Malloc fails for temp_localQ[]");
    for (j = 0; j < nrhs; ++j) {
        for (i = 0; i < local_ldx; ++i) {
            temp_localQ[j*local_ldx+i] = localX[j*local_ldx+i];
        }
    }

    LWORK_local = -1;
    if ( !(tau_local = doubleMalloc_dist(lengtau_local)) )
        ABORT("Malloc fails for tau_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, &WKOPT_local, &LWORK_local, &INFO_local);
    LWORK_local = (int) WKOPT_local;
    if ( !(WORK_local = doubleMalloc_dist(LWORK_local)) )
        ABORT("Malloc fails for WORK_local[]");
    dgeqrf_(&m_local, &n_local, temp_localQ, &m_local, tau_local, WORK_local, &LWORK_local, &INFO_local);

    rnum = m_local >= n_local ? nrhs*(nrhs+1)/2 : m_local*(nrhs+nrhs-m_local+1)/2;
    if ( !(localR = doubleMalloc_dist(rnum)) )
        ABORT("Malloc fails for localR[]");
    ct = 0;
    for (j = 0; j < nrhs; ++j) {
        k = j >= lengtau_local ? lengtau_local-1 : j;
        for (i = 0; i <= k; ++i) {
            localR[ct] = temp_localQ[j*local_ldx+i];
            ct++;
        }
    }

    MPI_Gather(&rnum, 1, MPI_INT, rnum_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        localR_disp[0] = 0;
        for (j = 1; j < nproc; ++j) {
            localR_disp[j] = localR_disp[j-1] + rnum_comb[j-1];
        }
        rnum_total = localR_disp[nproc-1] + rnum_comb[nproc-1];
    
        if ( !(gathered_localR = doubleMalloc_dist(rnum_total)) )
            ABORT("Malloc fails for gathered_localR[]");
    }
    MPI_Gatherv(localR, rnum, MPI_DOUBLE, gathered_localR, rnum_comb, localR_disp, MPI_DOUBLE, 0, grid->comm);
    MPI_Gather(&lengtau_local, 1, MPI_INT, m_local_comb, 1, MPI_INT, 0, grid->comm);
    if (!iam) {
        if ( !(gathered_localR = doubleMalloc_dist(nproc*rnum)) )
            ABORT("Malloc fails for gathered_localR[]");
    }
    MPI_Gather(localR, rnum, MPI_DOUBLE, gathered_localR, rnum, MPI_DOUBLE, 0, grid->comm);
    if (!iam) {
        m_local_disp[0] = 0;
        for (p = 1; p < nproc; ++p) {
            m_local_disp[p] = m_local_disp[p-1] + m_local_comb[p-1];
        }
        m_global = m_local_disp[nproc-1] + m_local_comb[nproc-1];

        if ( !(gathered_localR_mat = doubleMalloc_dist(m_global*nrhs)) )
            ABORT("Malloc fails for gathered_localR_mat[]");
        for (p = 0; p < nproc; ++p) {
            for (j = 0; j < nrhs; ++j) {
                offset = j < m_local_comb[p] ? j*(j+1)/2 : m_local_comb[p]*(m_local_comb[p]+1)/2+(j-m_local_comb[p])*m_local_comb[p];
                for (i = 0; i < m_local_comb[p]; ++i) {
                    gathered_localR_mat[j*m_global+m_local_disp[p]+i] = i <= j ? gathered_localR[localR_disp[p]+offset+i] : 0.0;
                }
            }
        }

        LWORK_global = -1;
        lengtau_global = m_global <= n_global ? m_global : n_global;
        if ( !(tau_global = doubleMalloc_dist(lengtau_global)) )
            ABORT("Malloc fails for tau_global[]");
        if ( !(localPerm = intMalloc_dist(nrhs)) )
            ABORT("Malloc fails for localPerm[]");
        for (j = 0; j < nrhs; ++j) {
            localPerm[j] = 0;
        }
        dgeqp3_(&m_global, &n_global, gathered_localR_mat, &m_global, localPerm, tau_global, &WKOPT_global, &LWORK_global, &INFO_global);
        LWORK_global = (int) WKOPT_global;
        if ( !(WORK_global = doubleMalloc_dist(LWORK_global)) )
            ABORT("Malloc fails for WORK_global[]");
        dgeqp3_(&m_global, &n_global, gathered_localR_mat, &m_global, localPerm, tau_global, WORK_global, &LWORK_global, &INFO_global);

        newr = 0;
        for (j = 0; j < lengtau_global; ++j) {
            // printf("%dth diagonal element is %f.\n", j+1, gathered_localR_mat[j*m_global+j]);
            if (fabs(gathered_localR_mat[j*m_global+j]) <= tol) {
                break;
            }
            newr++;
        }
        // printf("Truncated rank is %d.\n", newr);
        // fflush(stdout);
        MPI_Bcast(&newr, 1, MPI_INT, 0, grid->comm);

        SUPERLU_FREE(gathered_localR);
        SUPERLU_FREE(gathered_localR_mat);
        SUPERLU_FREE(tau_global);
        SUPERLU_FREE(WORK_global);
        SUPERLU_FREE(localPerm);
    }
    else {
        MPI_Bcast(&newr, 1, MPI_INT, 0, grid->comm);
    }

    *rank = newr;

    SUPERLU_FREE(temp_localQ);
    SUPERLU_FREE(tau_local);
    SUPERLU_FREE(WORK_local);
    SUPERLU_FREE(localR);
}

void dCPQR_dist_rand_getQ(double *localX, int local_ldx, double **localQ, int nrhs, int *rank, gridinfo_t *grid, double tol, int ovsamp)
{
    int_t i, j;
    int rand_col, rand_tot, opts = 3;
    int r1 = rand()%4096, r2 = rand()%4096, r3 = rand()%4096, r4 = rand()%4096;
    int iseed[4] = {r1, r2, r3, r4+(r4%2 == 0?1:0)};
    double *rand_mat, *local_X_rand, *R, *tmpQ;
    double one = 1.0, zero = 0.0;

    if (grid->iam != -1) {
        dCPQR_dist_getrank(localX, local_ldx, nrhs, rank, grid, tol);

        if (grid->iam == 0) {
            printf("Rank found is %d.\n", *rank);
            fflush(stdout);
        }
    }

    rand_col = *rank + ovsamp > nrhs ? nrhs : *rank + ovsamp;
    rand_tot = nrhs * rand_col;
    if ( !(rand_mat = doubleMalloc_dist(rand_tot)) )
        ABORT("Malloc fails for rand_mat[]");
    if ( !(local_X_rand = doubleMalloc_dist(local_ldx*rand_col)) )
        ABORT("Malloc fails for local_X_rand[]");
    if ( !(*localQ = doubleMalloc_dist(local_ldx*(*rank))) )
        ABORT("Malloc fails for localQ[]");

    if (grid->iam == 0) {
        dlarnv_(&opts, iseed, &rand_tot, rand_mat);
    }
    if (grid->iam != -1) {
        MPI_Bcast(rand_mat, rand_tot, MPI_DOUBLE, 0, grid->comm);

        printf("Grid %d in the grid gets random matrix with %d columns.\n", grid->iam, rand_col);
        fflush(stdout);

        dgemm_("N", "N", &local_ldx, &rand_col, &nrhs, &one, localX, &local_ldx, rand_mat, &nrhs, &zero, local_X_rand, &local_ldx);
        printf("Grid %d in the grid gets multiplication done.\n", grid->iam);
        fflush(stdout);

        dQR_dist(local_X_rand, local_ldx, &tmpQ, &R, rand_col, grid);
        printf("Grid %d in the grid gets QR.\n", grid->iam);
        fflush(stdout);

        for (j = 0; j < *rank; ++j) {
            for (i = 0; i < local_ldx; ++i) {
                (*localQ)[j*local_ldx+i] = tmpQ[j*local_ldx+i];
            }
        }

        SUPERLU_FREE(tmpQ);
        SUPERLU_FREE(R);
    }

    SUPERLU_FREE(local_X_rand);
    SUPERLU_FREE(rand_mat);
}

void dtruncated_SVD(double *X, int m, int n, int *r, double **U, double **S, double **V, double tol)
{
    int_t i, j;
    double *XX, *smallU, *smallS, *smallVT;
    double *WORK;
    int INFO, LWORK, newr;
    double WKOPT;
    int k = m <= n ? m : n;

    if ( !(XX = doubleMalloc_dist(m*n)) )
        ABORT("Malloc fails for XX[]");
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            XX[j*m+i] = X[j*m+i];
        }
    }

    LWORK = -1;
    if ( !(smallU = doubleMalloc_dist(m*m)) )
        ABORT("Malloc fails for smallU[]");
    if ( !(smallS = doubleMalloc_dist(k)) )
        ABORT("Malloc fails for smallS[]");
    if ( !(smallVT = doubleMalloc_dist(n*n)) )
        ABORT("Malloc fails for smallVT[]");
    dgesvd_("All", "All", &m, &n, XX, &m, smallS, smallU, &m, smallVT, &n, &WKOPT, &LWORK, &INFO);

    LWORK = (int) WKOPT;
    if ( !(WORK = doubleMalloc_dist(LWORK)) )
        ABORT("Malloc fails for WORK[]");
    dgesvd_("All", "All", &m, &n, XX, &m, smallS, smallU, &m, smallVT, &n, WORK, &LWORK, &INFO);

    newr = 0;
    for (j = 0; j < k; ++j) {
        if (smallS[j] <= smallS[0]*tol) {
            break;
        }
        newr++;
    }

    if ( !(*U = doubleMalloc_dist(m*newr)) )
        ABORT("Malloc fails for *U[]");
    if ( !(*S = doubleMalloc_dist(newr)) )
        ABORT("Malloc fails for *S[]");
    if ( !(*V = doubleMalloc_dist(n*newr)) )
        ABORT("Malloc fails for *V[]");
    for (j = 0; j < newr; ++j) {
        (*S)[j] = smallS[j];
        for (i = 0; i < m; ++i) {
            (*U)[j*m+i] = smallU[j*m+i];
        }
        for (i = 0; i < n; ++i) {
            (*V)[j*n+i] = smallVT[i*n+j];
        }
    }

    *r = newr;

    SUPERLU_FREE(XX);
    SUPERLU_FREE(smallU);
    SUPERLU_FREE(smallS);
    SUPERLU_FREE(smallVT);
    SUPERLU_FREE(WORK);
}

void drecompression_dist(double *localX, int local_ldx, int global_ldx, double **localU, double *D, double **S, gridinfo_t *grid_A,
    double *localY, int local_ldy, int global_ldy, double **localV, gridinfo_t *grid_B, int *nrhs, double tol)
{
    int_t i, j;
    double *R_A, *R_B, *rR_B, *R_A_R_B;
    double *localQ_A, *localQ_B;
    double *smallU, *smallS, *smallVT, *truncateU, *truncateV, *r_truncateV;
    double *WORK;
    int INFO, LWORK;
    int iam_A = grid_A->iam, iam_B = grid_B->iam;
    int nproc_A = grid_A->nprow*grid_A->npcol, nproc_B = grid_B->nprow*grid_B->npcol;
    int local_ldx_all[nproc_A], local_ldy_all[nproc_B];
    int r = *nrhs, newr, offset;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    double one = 1.0, zero = 0.0, WKOPT;

    if ((global_ldx > r) && (global_ldy > r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A != -1) {
            dQR_dist(localX, local_ldx, &localQ_A, &R_A, r, grid_A);
        }
        if (iam_B != -1) {
            dQR_dist(localY, local_ldy, &localQ_B, &R_B, r, grid_B);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        transfer_X(R_B, r, r, rR_B, grid_A, 0);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < r; ++i) {
                    R_A[j*r+i] = R_A[j*r+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &r, &r, &r, &one, R_A, &r, rR_B, &r, &zero, R_A_R_B, &r);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(r)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &r, &r, R_A_R_B, &r, smallS, smallU, &r, smallVT, &r, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &r, &r, R_A_R_B, &r, smallS, smallU, &r, smallVT, &r, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < r; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < r; ++i) {
                    truncateU[j*r+i] = smallU[j*r+i];
                    truncateV[j*r+i] = smallVT[i*r+j];
                }
            }
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        if (!global_rank) {
            MPI_Send(&newr, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == nproc_A) {
            MPI_Recv(&newr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X(truncateV, r, newr, r_truncateV, grid_A, 1);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            dgemm_("N", "N", &local_ldx, &newr, &r, &one, localQ_A, &local_ldx, truncateU, &r, &zero, *localU, &local_ldx);

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldy, &newr, &r, &one, localQ_B, &local_ldy, r_truncateV, &r, &zero, *localV, &local_ldy);

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else if ((global_ldx <= r) && (global_ldy > r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A == 0) {
            if ( !(R_A = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A[]");
        }
        if (iam_A != -1) {
            dgather_X(localX, local_ldx, R_A, global_ldx, r, grid_A);
        }
        if (iam_B != -1) {
            dQR_dist(localY, local_ldy, &localQ_B, &R_B, r, grid_B);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        transfer_X(R_B, r, r, rR_B, grid_A, 0);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < global_ldx; ++i) {
                    R_A[j*global_ldx+i] = R_A[j*global_ldx+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &global_ldx, &r, &r, &one, R_A, &global_ldx, rR_B, &r, &zero, R_A_R_B, &global_ldx);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(global_ldx*global_ldx)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(global_ldx)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &global_ldx, &r, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &r, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &global_ldx, &r, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &r, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < global_ldx; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < global_ldx; ++i) {
                    truncateU[j*global_ldx+i] = smallU[j*global_ldx+i];
                }
                for (i = 0; i < r; ++i) {
                    truncateV[j*r+i] = smallVT[i*r+j];
                }
            }
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        if (!global_rank) {
            MPI_Send(&newr, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == nproc_A) {
            MPI_Recv(&newr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X(truncateV, r, newr, r_truncateV, grid_A, 1);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            MPI_Allgather(&local_ldx, 1, MPI_INT, local_ldx_all, 1, MPI_INT, grid_A->comm);

            offset = 0;
            for (j = 0; j < iam_A; ++j) {
                offset += local_ldx_all[j];
            }
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldx; ++i) {
                    (*localU)[j*local_ldx+i] = truncateU[j*global_ldx+offset+i];
                }
            }

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldy, &newr, &r, &one, localQ_B, &local_ldy, r_truncateV, &r, &zero, *localV, &local_ldy);

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else if ((global_ldx > r) && (global_ldy <= r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_B == 0) {
            if ( !(R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_B[]");
        }
        if (iam_A != -1) {
            dQR_dist(localX, local_ldx, &localQ_A, &R_A, r, grid_A);
        }
        if (iam_B != -1) {
            dgather_X(localY, local_ldy, R_B, global_ldy, r, grid_B);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        transfer_X(R_B, global_ldy, r, rR_B, grid_A, 0);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < r; ++i) {
                    R_A[j*r+i] = R_A[j*r+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &r, &global_ldy, &r, &one, R_A, &r, rR_B, &global_ldy, &zero, R_A_R_B, &r);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(global_ldy)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(global_ldy*global_ldy)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &r, &global_ldy, R_A_R_B, &r, smallS, smallU, &r, smallVT, &global_ldy, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &r, &global_ldy, R_A_R_B, &r, smallS, smallU, &r, smallVT, &global_ldy, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < global_ldy; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < r; ++i) {
                    truncateU[j*r+i] = smallU[j*r+i];
                }
                for (i = 0; i < global_ldy; ++i) {
                    truncateV[j*global_ldy+i] = smallVT[i*global_ldy+j];
                }
            }
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        if (!global_rank) {
            MPI_Send(&newr, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == nproc_A) {
            MPI_Recv(&newr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X(truncateV, global_ldy, newr, r_truncateV, grid_A, 1);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldx, &newr, &r, &one, localQ_A, &local_ldx, truncateU, &r, &zero, *localU, &local_ldx);

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            MPI_Allgather(&local_ldy, 1, MPI_INT, local_ldy_all, 1, MPI_INT, grid_B->comm);

            offset = 0;
            for (j = 0; j < iam_B; ++j) {
                offset += local_ldy_all[j];
            }
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldy; ++i) {
                    (*localV)[j*local_ldy+i] = r_truncateV[j*global_ldy+offset+i];
                }
            }

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A == 0) {
            if ( !(R_A = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A[]");
        }
        if (iam_A != -1) {
            dgather_X(localX, local_ldx, R_A, global_ldx, r, grid_A);
        }

        if (iam_B == 0) {
            if ( !(R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_B[]");
        }
        if (iam_B != -1) {
            dgather_X(localY, local_ldy, R_B, global_ldy, r, grid_B);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        transfer_X(R_B, global_ldy, r, rR_B, grid_A, 0);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < global_ldx; ++i) {
                    R_A[j*global_ldx+i] = R_A[j*global_ldx+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldy*global_ldx)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &global_ldx, &global_ldy, &r, &one, R_A, &global_ldx, rR_B, &global_ldy, &zero, R_A_R_B, &global_ldx);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(global_ldx*global_ldx)) )
                ABORT("Malloc fails for smallU[]");
            int xy = global_ldx <= global_ldy ? global_ldx : global_ldy;
            if ( !(smallS = doubleMalloc_dist(xy)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(global_ldy*global_ldy)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &global_ldx, &global_ldy, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &global_ldy, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &global_ldx, &global_ldy, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &global_ldy, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < xy; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < global_ldx; ++i) {
                    truncateU[j*global_ldx+i] = smallU[j*global_ldx+i];
                }
                for (i = 0; i < global_ldy; ++i) {
                    truncateV[j*global_ldy+i] = smallVT[i*global_ldy+j];
                }
            }
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        if (!global_rank) {
            MPI_Send(&newr, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == nproc_A) {
            MPI_Recv(&newr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X(truncateV, global_ldy, newr, r_truncateV, grid_A, 1);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            MPI_Allgather(&local_ldx, 1, MPI_INT, local_ldx_all, 1, MPI_INT, grid_A->comm);

            offset = 0;
            for (j = 0; j < iam_A; ++j) {
                offset += local_ldx_all[j];
            }
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldx; ++i) {
                    (*localU)[j*local_ldx+i] = truncateU[j*global_ldx+offset+i];
                }
            }

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            MPI_Allgather(&local_ldy, 1, MPI_INT, local_ldy_all, 1, MPI_INT, grid_B->comm);

            offset = 0;
            for (j = 0; j < iam_B; ++j) {
                offset += local_ldy_all[j];
            }
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldy; ++i) {
                    (*localV)[j*local_ldy+i] = r_truncateV[j*global_ldy+offset+i];
                }
            }

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }    
}

void drecompression_dist_twogrids(double *localX, int local_ldx, int global_ldx, double **localU, double *D, double **S, gridinfo_t *grid_A,
    double *localY, int local_ldy, int global_ldy, double **localV, gridinfo_t *grid_B, int *nrhs, double tol, int *grid_proc, int grA, int grB)
{
    int_t i, j;
    double *R_A, *R_B, *rR_B, *R_A_R_B;
    double *localQ_A, *localQ_B;
    double *smallU, *smallS, *smallVT, *truncateU, *truncateV, *r_truncateV;
    double *WORK;
    int INFO, LWORK;
    int iam_A = grid_A->iam, iam_B = grid_B->iam;
    int nproc_A = grid_A->nprow*grid_A->npcol, nproc_B = grid_B->nprow*grid_B->npcol;
    int local_ldx_all[nproc_A], local_ldy_all[nproc_B];
    int r = *nrhs, newr, offset;
    int global_rank, globalA = 0, globalB = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    double one = 1.0, zero = 0.0, WKOPT;

    if ((global_ldx > r) && (global_ldy > r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A != -1) {
            dQR_dist(localX, local_ldx, &localQ_A, &R_A, r, grid_A);
            MPI_Barrier(grid_A->comm);
        }
        if (iam_B != -1) {
            dQR_dist(localY, local_ldy, &localQ_B, &R_B, r, grid_B);
            MPI_Barrier(grid_B->comm);
        }

        transfer_X_dgrids(R_B, r, r, rR_B, grid_proc, grB, grA);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < r; ++i) {
                    R_A[j*r+i] = R_A[j*r+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &r, &r, &r, &one, R_A, &r, rR_B, &r, &zero, R_A_R_B, &r);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(r)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &r, &r, R_A_R_B, &r, smallS, smallU, &r, smallVT, &r, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &r, &r, R_A_R_B, &r, smallS, smallU, &r, smallVT, &r, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < r; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < r; ++i) {
                    truncateU[j*r+i] = smallU[j*r+i];
                    truncateV[j*r+i] = smallVT[i*r+j];
                }
            }
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        for (j = 0; j < grA; ++j) {
            globalA += grid_proc[j];
        }
        for (j = 0; j < grB; ++j) {
            globalB += grid_proc[j];
        }

        if (global_rank == globalA) {
            MPI_Send(&newr, 1, MPI_INT, globalB, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == globalB) {
            MPI_Recv(&newr, 1, MPI_INT, globalA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X_dgrids(truncateV, r, newr, r_truncateV, grid_proc, grA, grB);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            dgemm_("N", "N", &local_ldx, &newr, &r, &one, localQ_A, &local_ldx, truncateU, &r, &zero, *localU, &local_ldx);

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldy, &newr, &r, &one, localQ_B, &local_ldy, r_truncateV, &r, &zero, *localV, &local_ldy);

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else if ((global_ldx <= r) && (global_ldy > r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A == 0) {
            if ( !(R_A = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A[]");
        }
        if (iam_A != -1) {
            dgather_X(localX, local_ldx, R_A, global_ldx, r, grid_A);
            MPI_Barrier(grid_A->comm);
        }
        if (iam_B != -1) {
            dQR_dist(localY, local_ldy, &localQ_B, &R_B, r, grid_B);
            MPI_Barrier(grid_B->comm);
        }

        transfer_X_dgrids(R_B, r, r, rR_B, grid_proc, grB, grA);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < global_ldx; ++i) {
                    R_A[j*global_ldx+i] = R_A[j*global_ldx+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &global_ldx, &r, &r, &one, R_A, &global_ldx, rR_B, &r, &zero, R_A_R_B, &global_ldx);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(global_ldx*global_ldx)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(global_ldx)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &global_ldx, &r, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &r, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &global_ldx, &r, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &r, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < global_ldx; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < global_ldx; ++i) {
                    truncateU[j*global_ldx+i] = smallU[j*global_ldx+i];
                }
                for (i = 0; i < r; ++i) {
                    truncateV[j*r+i] = smallVT[i*r+j];
                }
            }
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        for (j = 0; j < grA; ++j) {
            globalA += grid_proc[j];
        }
        for (j = 0; j < grB; ++j) {
            globalB += grid_proc[j];
        }

        if (global_rank == globalA) {
            MPI_Send(&newr, 1, MPI_INT, globalB, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == globalB) {
            MPI_Recv(&newr, 1, MPI_INT, globalA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X_dgrids(truncateV, r, newr, r_truncateV, grid_proc, grA, grB);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, r*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            MPI_Allgather(&local_ldx, 1, MPI_INT, local_ldx_all, 1, MPI_INT, grid_A->comm);

            offset = 0;
            for (j = 0; j < iam_A; ++j) {
                offset += local_ldx_all[j];
            }
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldx; ++i) {
                    (*localU)[j*local_ldx+i] = truncateU[j*global_ldx+offset+i];
                }
            }

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldy, &newr, &r, &one, localQ_B, &local_ldy, r_truncateV, &r, &zero, *localV, &local_ldy);

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else if ((global_ldx > r) && (global_ldy <= r)) {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_B == 0) {
            if ( !(R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_B[]");
        }
        if (iam_A != -1) {
            dQR_dist(localX, local_ldx, &localQ_A, &R_A, r, grid_A);
            MPI_Barrier(grid_A->comm);
        }
        if (iam_B != -1) {
            dgather_X(localY, local_ldy, R_B, global_ldy, r, grid_B);
            MPI_Barrier(grid_B->comm);
        }

        transfer_X_dgrids(R_B, global_ldy, r, rR_B, grid_proc, grB, grA);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < r; ++i) {
                    R_A[j*r+i] = R_A[j*r+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &r, &global_ldy, &r, &one, R_A, &r, rR_B, &global_ldy, &zero, R_A_R_B, &r);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(r*r)) )
                ABORT("Malloc fails for smallU[]");
            if ( !(smallS = doubleMalloc_dist(global_ldy)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(global_ldy*global_ldy)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &r, &global_ldy, R_A_R_B, &r, smallS, smallU, &r, smallVT, &global_ldy, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &r, &global_ldy, R_A_R_B, &r, smallS, smallU, &r, smallVT, &global_ldy, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < global_ldy; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < r; ++i) {
                    truncateU[j*r+i] = smallU[j*r+i];
                }
                for (i = 0; i < global_ldy; ++i) {
                    truncateV[j*global_ldy+i] = smallVT[i*global_ldy+j];
                }
            }
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(r*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, r*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        for (j = 0; j < grA; ++j) {
            globalA += grid_proc[j];
        }
        for (j = 0; j < grB; ++j) {
            globalB += grid_proc[j];
        }

        if (global_rank == globalA) {
            MPI_Send(&newr, 1, MPI_INT, globalB, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == globalB) {
            MPI_Recv(&newr, 1, MPI_INT, globalA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X_dgrids(truncateV, global_ldy, newr, r_truncateV, grid_proc, grA, grB);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localV[]");
            dgemm_("N", "N", &local_ldx, &newr, &r, &one, localQ_A, &local_ldx, truncateU, &r, &zero, *localU, &local_ldx);

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            MPI_Allgather(&local_ldy, 1, MPI_INT, local_ldy_all, 1, MPI_INT, grid_B->comm);

            offset = 0;
            for (j = 0; j < iam_B; ++j) {
                offset += local_ldy_all[j];
            }
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldy; ++i) {
                    (*localV)[j*local_ldy+i] = r_truncateV[j*global_ldy+offset+i];
                }
            }

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
    else {
        if (!iam_A) {
            if ( !(rR_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for rR_B[]");
        }

        if (iam_A == 0) {
            if ( !(R_A = doubleMalloc_dist(global_ldx*r)) )
                ABORT("Malloc fails for R_A[]");
        }
        if (iam_A != -1) {
            dgather_X(localX, local_ldx, R_A, global_ldx, r, grid_A);
            MPI_Barrier(grid_A->comm);
        }

        if (iam_B == 0) {
            if ( !(R_B = doubleMalloc_dist(global_ldy*r)) )
                ABORT("Malloc fails for R_B[]");
        }
        if (iam_B != -1) {
            dgather_X(localY, local_ldy, R_B, global_ldy, r, grid_B);
            MPI_Barrier(grid_B->comm);
        }

        transfer_X_dgrids(R_B, global_ldy, r, rR_B, grid_proc, grB, grA);

        if (!iam_A) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < global_ldx; ++i) {
                    R_A[j*global_ldx+i] = R_A[j*global_ldx+i]*D[j];
                }
            }
            if ( !(R_A_R_B = doubleMalloc_dist(global_ldy*global_ldx)) )
                ABORT("Malloc fails for R_A_R_B[]");
            dgemm_("N", "T", &global_ldx, &global_ldy, &r, &one, R_A, &global_ldx, rR_B, &global_ldy, &zero, R_A_R_B, &global_ldx);

            LWORK = -1;
            if ( !(smallU = doubleMalloc_dist(global_ldx*global_ldx)) )
                ABORT("Malloc fails for smallU[]");
            int xy = global_ldx <= global_ldy ? global_ldx : global_ldy;
            if ( !(smallS = doubleMalloc_dist(xy)) )
                ABORT("Malloc fails for smallS[]");
            if ( !(smallVT = doubleMalloc_dist(global_ldy*global_ldy)) )
                ABORT("Malloc fails for smallVT[]");
            dgesvd_("All", "All", &global_ldx, &global_ldy, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &global_ldy, &WKOPT, &LWORK, &INFO);

            LWORK = (int) WKOPT;
            if ( !(WORK = doubleMalloc_dist(LWORK)) )
                ABORT("Malloc fails for WORK[]");
            dgesvd_("All", "All", &global_ldx, &global_ldy, R_A_R_B, &global_ldx, smallS, smallU, &global_ldx, smallVT, &global_ldy, WORK, &LWORK, &INFO);

            newr = 0;
            for (j = 0; j < xy; ++j) {
                if (smallS[j] <= smallS[0]*tol) {
                    break;
                }
                newr++;
            }

            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            if ( !(*S = doubleMalloc_dist(newr)) )
                ABORT("Malloc fails for *S[]");
            if ( !(truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for truncateV[]");
            for (j = 0; j < newr; ++j) {
                (*S)[j] = smallS[j];
                for (i = 0; i < global_ldx; ++i) {
                    truncateU[j*global_ldx+i] = smallU[j*global_ldx+i];
                }
                for (i = 0; i < global_ldy; ++i) {
                    truncateV[j*global_ldy+i] = smallVT[i*global_ldy+j];
                }
            }
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);

            // printf("Truncated SVD done.\n");
            // fflush(stdout);

            SUPERLU_FREE(R_A);
            SUPERLU_FREE(rR_B);
            SUPERLU_FREE(R_A_R_B);
            SUPERLU_FREE(WORK);
            SUPERLU_FREE(smallU);
            SUPERLU_FREE(smallS);
            SUPERLU_FREE(smallVT);
        }
        else if (iam_A != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_A->comm);

            if ( !(truncateU = doubleMalloc_dist(global_ldx*newr)) )
                ABORT("Malloc fails for truncateU[]");
            MPI_Bcast(truncateU, global_ldx*newr, MPI_DOUBLE, 0, grid_A->comm);
        }

        for (j = 0; j < grA; ++j) {
            globalA += grid_proc[j];
        }
        for (j = 0; j < grB; ++j) {
            globalB += grid_proc[j];
        }

        if (global_rank == globalA) {
            MPI_Send(&newr, 1, MPI_INT, globalB, 0, MPI_COMM_WORLD);
        }
        else if (global_rank == globalB) {
            MPI_Recv(&newr, 1, MPI_INT, globalA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (!iam_B) {
            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");

            SUPERLU_FREE(R_B);
        }
        transfer_X_dgrids(truncateV, global_ldy, newr, r_truncateV, grid_proc, grA, grB);

        if (!iam_B) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }
        else if (iam_B != -1) {
            MPI_Bcast(&newr, 1, MPI_INT, 0, grid_B->comm);

            if ( !(r_truncateV = doubleMalloc_dist(global_ldy*newr)) )
                ABORT("Malloc fails for r_truncateV[]");
            MPI_Bcast(r_truncateV, global_ldy*newr, MPI_DOUBLE, 0, grid_B->comm);
        }

        if (iam_A != -1) {
            MPI_Allgather(&local_ldx, 1, MPI_INT, local_ldx_all, 1, MPI_INT, grid_A->comm);

            offset = 0;
            for (j = 0; j < iam_A; ++j) {
                offset += local_ldx_all[j];
            }
            if ( !(*localU = doubleMalloc_dist(local_ldx*newr)) )
                ABORT("Malloc fails for *localU[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldx; ++i) {
                    (*localU)[j*local_ldx+i] = truncateU[j*global_ldx+offset+i];
                }
            }

            SUPERLU_FREE(truncateU);
        }
        if (iam_B != -1) {
            MPI_Allgather(&local_ldy, 1, MPI_INT, local_ldy_all, 1, MPI_INT, grid_B->comm);

            offset = 0;
            for (j = 0; j < iam_B; ++j) {
                offset += local_ldy_all[j];
            }
            if ( !(*localV = doubleMalloc_dist(local_ldy*newr)) )
                ABORT("Malloc fails for *localV[]");
            for (j = 0; j < newr; ++j) {
                for (i = 0; i < local_ldy; ++i) {
                    (*localV)[j*local_ldy+i] = r_truncateV[j*global_ldy+offset+i];
                }
            }

            SUPERLU_FREE(r_truncateV);
        }

        *nrhs = newr;

        if (!iam_A) {
            SUPERLU_FREE(truncateV);
        }
    }
}

void dcheck_error_fadi(int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    double *F, int r, double *Z, double *D, double *Y, int rank, double *trueX)
{
    SuperMatrix GA, GB;
    char     transpose[1];
    *transpose = 'N';
    double *ZD, *rY;
    double *X, *X_transpose, *rX, *AX, *BXT, *rBXT, *RHS;
    int iam_A = grid_A->iam;
    int iam_B = grid_B->iam;
    int_t i, j;
    double err1 = 0.0, err2 = 0.0, norm1 = 0.0, norm2 = 0.0;
    double s;
    double one = 1.0, zero = 0.0;

    if (iam_A == 0) {
        if ( !(rY = doubleMalloc_dist(m_B*rank)) )
            ABORT("Malloc fails for rY");
    }
    transfer_X(Y, m_B, rank, rY, grid_A, 0);

    // if (iam_A == 0) {
    //     printf("Y on grid_A root is\n");
    //     for (i = 0; i < m_B; ++i) {
    //         for (j = 0; j < rank; ++j) {
    //             printf("%f ",rY[j*m_B+i]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }

    if (iam_A == 0) {
        if ( !(ZD = doubleMalloc_dist(m_A*rank)) )
            ABORT("Malloc fails for ZD");
        for (j = 0; j < rank; ++j) {
            for (i = 0; i < m_A; ++i) {
                ZD[j*m_A+i] = Z[j*m_A+i]*D[j];
            }
        }

        if ( !(X = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for X");
        dgemm_("N", "T", &m_A, &m_B, &rank, &one, ZD, &m_A, rY, &m_B, &zero, X, &m_A);
    
        /* Create compressed column matrix for GA. */
        dCreate_CompCol_Matrix_dist(&GA, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A,
            SLU_NC, SLU_D, SLU_GE);

        if ( !(AX = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for AX[]");
        sp_dgemm_dist(transpose, m_B, 1.0, &GA, X, m_A, 0.0, AX, m_A);

        /* Destroy GA */
        Destroy_CompCol_Matrix_dist(&GA);

        SUPERLU_FREE(ZD);
        SUPERLU_FREE(rY);
    }

    if (iam_B == 0) {
        if ( !(rX = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for rX");
    }
    transfer_X(X, m_A, m_B, rX, grid_A, 1);

    if (iam_B == 0) {
        /* Create compressed column matrix for GA. */
        dCreate_CompCol_Matrix_dist(&GB, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
            SLU_NC, SLU_D, SLU_GE);

        if ( !(X_transpose = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for X_transpose[]");
        if ( !(BXT = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for BXT[]");


        for (j = 0; j < m_A; ++j) {
            for (i = 0; i < m_B; ++i) {
                X_transpose[j*m_B+i] = rX[i*m_A+j];
            }
        }
        sp_dgemm_dist(transpose, m_A, 1.0, &GB, X_transpose, m_B, 0.0, BXT, m_B);

        /* Destroy GA */
        Destroy_CompCol_Matrix_dist(&GB);
    }

    if (iam_A == 0) {
        if ( !(rBXT = doubleMalloc_dist(m_A*m_B)) )
            ABORT("Malloc fails for rBXT[]");
    }
    transfer_X(BXT, m_B, m_A, rBXT, grid_A, 0);

    if (iam_A == 0) {
        for (j = 0; j < m_B; ++j) {
            for (i = 0; i < m_A; ++i) {
                s = F[j*m_A+i] - (AX[j*m_A+i] - rBXT[i*m_B+j]);
                err1 += s*s;

                s = X[j*m_A+i] - trueX[j*m_A+i];
                err2 += s*s;

                s = F[j*m_A+i];
                norm1 += s*s;

                s = trueX[j*m_A+i];
                norm2 += s*s;
            }
        }
        err1 = sqrt(err1) / sqrt(norm1);
        err2 = sqrt(err2) / sqrt(norm2);
        printf("Relative error of approximating RHS is %f, and approximating true solution is %f\n.", err1, err2);
        fflush(stdout);

        SUPERLU_FREE(X);
        SUPERLU_FREE(rBXT);
        SUPERLU_FREE(AX);

    }

    if (iam_B == 0) {
        SUPERLU_FREE(BXT);
        SUPERLU_FREE(X_transpose);
        SUPERLU_FREE(rX);
    }
}

void ellipj(double *u, double *dn, double m, int_t l)
{
    double *a, *b, *c;
    double tol = 2.22*pow(10.0, -16.0);
    int_t len = 1000, iter = 0, i, j;
    double *phi;

    if ( !(a = doubleMalloc_dist(len)) )
        ABORT("Malloc fails for a[]");
    if ( !(b = doubleMalloc_dist(len)) )
        ABORT("Malloc fails for b[]");
    if ( !(c = doubleMalloc_dist(len)) )
        ABORT("Malloc fails for c[]");
    a[0] = 1.0;
    b[0] = sqrt(1.0-m);
    c[0] = sqrt(m);
    while (c[iter] > tol) {
        a[iter+1] = (a[iter] + b[iter]) / 2.0;
        b[iter+1] = sqrt(a[iter] * b[iter]);
        c[iter+1] = (a[iter] - b[iter]) / 2.0;
        iter++;
    }

    for (i = 0; i < l; ++i) {
        if ( !(phi = doubleMalloc_dist(iter+1)) )
            ABORT("Malloc fails for phi[]");
        phi[iter] = pow(2.0, (double) iter) * a[iter] * u[i];
        for (j = iter-1; j >= 0; --j) {
            phi[j] = (phi[j+1] + asin(c[j+1] / a[j+1] * sin(phi[j+1]))) / 2.0;
        }
        dn[i] = cos(phi[0]) / cos(phi[1] - phi[0]);
    }
}

double ellipke(double k)
{
    double c1 = 1.0/24.0, c2 = 3.0/44.0, c3 = 1.0/14.0;
    double tol = pow(pow(10.0, -16.0)*2.22*4.0,1.0/6.0);
    double xn = 0.0, yn = 1.0-pow(k, 2.0), zn = 1.0;
    double mu, lambda, epsilon, e2, e3, result, s;
    double xndev, yndev, zndev, xnroot, ynroot, znroot;
    
    while (1)
    {
        mu = (xn+yn+zn)/3.0;
        xndev = 2.0-(mu+xn)/mu;
        yndev = 2.0-(mu+yn)/mu;
        zndev = 2.0-(mu+zn)/mu;
        epsilon = MAX3(fabs(xndev),fabs(yndev),fabs(zndev));
        if (epsilon < tol) break;

        xnroot = sqrt(xn);
        ynroot = sqrt(yn);
        znroot = sqrt(zn);
        lambda = xnroot*(ynroot+znroot) +ynroot*znroot;

        xn = (xn+lambda)*0.25;
        yn = (yn+lambda)*0.25;
        zn = (zn+lambda)*0.25;
    }
    e2 = xndev*yndev - pow(zndev,2.0);
    e3 = xndev*yndev*zndev;

    s = 1.0 + (c1*e2-0.1-c2*e3)*e2 + c3*e3; 
    result = s/sqrt(mu);

    return result;
}

void getshifts_adi(double a, double b, double c, double d, double **p, double **q, int_t *l, double tol)
{
    double cr, gam, A, B, C, D;
    double dN, K, m1, kp, p1;
    int N;
    double *u, *dn;
    int_t j;

    cr = fabs((c-a) * (d-b) / ((c-b) * fabs(d-a))); 
    gam = -1.0 + 2.0 * cr + 2.0 * sqrt(pow(cr, 2.0) - cr); 
    A = -gam * a * (1.0 - gam) + gam * (c - gam * d) + c * gam - gam * d; 
    B = -gam * a * (c * gam - d) - a * (c * gam - gam * d) - gam * (c * d - gam * d * c); 
    C = a * (1.0 - gam) + gam * (c - d) + c * gam - d; 
    D = -gam * a * (c - d) - a * (c - gam * d) + c * d - gam * d * c;

    dN = 1.0 / pow(M_PI, 2.0) * log(4.0 / tol) * log(16.0 * cr);
    N = (int) (dN+1);
    *l = N;

    if ( !(u = doubleMalloc_dist(N)) )
        ABORT("Malloc fails for u[]");
    if ( !(dn = doubleMalloc_dist(N)) )
        ABORT("Malloc fails for dn[]");
    if (gam > pow(10.0, 7.0)) {
        K = (2.0*log(2.0)+log(gam)) + (-1.0+2.0*log(2.0)+log(gam))/pow(gam, 2.0)/4.0;
        m1 = 1/pow(gam, 2.0);
        for (j = 0; j < N; ++j) {
            u[j] = j + 0.5;
            dn[j] = 1.0/cosh(u[j]) + 0.25*m1*(sinh(u[j])*cosh(u[j])+u[j])*tanh(u[j])/cosh(u[j]);
        }
    }
    else {
        kp = 1.0-pow(1.0/gam, 2.0); 
        K = ellipke(kp);
        for (j = 0; j < N; ++j) {
            u[j] = (2.0*j+1.0)*K/2.0/N;
        }
        ellipj(u, dn, kp, N);
    }

    if ( !(*p = doubleMalloc_dist(N)) )
        ABORT("Malloc fails for *p[]");
    if ( !(*q = doubleMalloc_dist(N)) )
        ABORT("Malloc fails for *q[]");
    for (j = 0; j < N; ++j) {
        p1 = gam*dn[j];
        (*p)[j] = (-D*p1-B) / (C*p1+A);
        (*q)[j] = (D*p1-B) / (-C*p1+A);
    }
}

void dgenerate_shifts(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B)
{
    int_t i;
    int nproc_A = grid_A->nprow * grid_A->npcol;
    double *rp, *rq;
    double tol = pow(10.0, -10.0);
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (grid_A->iam == 0) {
        getshifts_adi(a, b, c, d, p, q, l, tol);

        MPI_Bcast(l, 1, MPI_INT, 0, grid_A->comm);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_A->comm);
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_A->comm);
    }

    if (!global_rank) {
        MPI_Send(l, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == nproc_A) {
        MPI_Recv(l, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid_B->iam == 0) {
        rp = (double *) doubleMalloc_dist(*l);
        rq = (double *) doubleMalloc_dist(*l);
    }
    transfer_X(*p, *l, 1, rp, grid_A, 1);
    transfer_X(*q, *l, 1, rq, grid_A, 1);

    if (grid_B->iam == 0) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);
        for (i = 0; i < *l; ++i) {
            (*p)[i] = rp[i];
            (*q)[i] = rq[i];
        }

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_B->comm);
    }
}

void dgenerate_shifts_onegrid(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid)
{
    int_t i;
    int nproc = grid->nprow * grid->npcol;
    int iam = grid->iam;
    double tol = pow(10.0, -10.0);

    if (iam == 0) {
        getshifts_adi(a, b, c, d, p, q, l, tol);

        MPI_Bcast(l, 1, MPI_INT, 0, grid->comm);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid->comm);
    }
    else {
        MPI_Bcast(l, 1, MPI_INT, 0, grid->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid->comm);
    }
}

void dgenerate_shifts_twogrids(double a, double b, double c, double d, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int *grid_proc, int grA, int grB)
{
    int_t i;
    double *rp, *rq;
    double tol = pow(10.0, -10.0);
    int global_rank, rootA = 0, rootB = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (grid_A->iam == 0) {
        getshifts_adi(a, b, c, d, p, q, l, tol);

        MPI_Bcast(l, 1, MPI_INT, 0, grid_A->comm);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_A->comm);
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_A->comm);
    }

    for (i = 0; i < grA; ++i) {
        rootA += grid_proc[i];
    }
    for (i = 0; i < grB; ++i) {
        rootB += grid_proc[i];
    }

    if (global_rank == rootA) {
        MPI_Send(l, 1, MPI_INT, rootB, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == rootB) {
        MPI_Recv(l, 1, MPI_INT, rootA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid_B->iam == 0) {
        rp = (double *) doubleMalloc_dist(*l);
        rq = (double *) doubleMalloc_dist(*l);
    }
    transfer_X_dgrids(*p, *l, 1, rp, grid_proc, grA, grB);
    transfer_X_dgrids(*q, *l, 1, rq, grid_proc, grA, grB);

    if (grid_B->iam == 0) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);
        for (i = 0; i < *l; ++i) {
            (*p)[i] = rp[i];
            (*q)[i] = rq[i];
        }

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(l, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(*l);
        *q = (double *) doubleMalloc_dist(*l);

        MPI_Bcast(*p, *l, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, *l, MPI_DOUBLE, 0, grid_B->comm);
    }
}

void dgather_TTcores(double **TTcores, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, double **TTcores_global)
{
    int_t k;
    int *aug_rs;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    for (k = 0; k < d; ++k) {
        if (grids[k]->iam == 0) {
            if ( !(TTcores_global[k] = doubleMalloc_dist(aug_rs[k]*ms[k]*aug_rs[k+1])) )
                ABORT("Malloc fails for global_T1[]");
        }
        if (grids[k]->iam != -1) {
            dgather_X(TTcores[k], aug_rs[k]*locals[k], TTcores_global[k], aug_rs[k]*ms[k], aug_rs[k+1], grids[k]);
        }
    }

    SUPERLU_FREE(aug_rs);
}

void dgather_TTcores_2grids(double **TTcores, gridinfo_t *grid1, gridinfo_t *grid2, int *ms, int *rs, int *locals, int d, double **TTcores_global)
{
    int_t k;
    int *aug_rs;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    for (k = 0; k < d-1; ++k) {
        if (grid1->iam == 0) {
            if ( !(TTcores_global[k] = doubleMalloc_dist(aug_rs[k]*ms[k]*aug_rs[k+1])) )
                ABORT("Malloc fails for TTcores_global[k][]");
        }
        if (grid1->iam != -1) {
            dgather_X(TTcores[k], aug_rs[k]*locals[k], TTcores_global[k], aug_rs[k]*ms[k], aug_rs[k+1], grid1);
        }
    }

    if (grid2->iam == 0) {
        if ( !(TTcores_global[d-1] = doubleMalloc_dist(aug_rs[d-1]*ms[d-1]*aug_rs[d])) )
            ABORT("Malloc fails for TTcores_global[d-1][]");
    }
    if (grid2->iam != -1) {
        dgather_X(TTcores[d-1], aug_rs[d-1]*locals[d-1], TTcores_global[d-1], aug_rs[d-1]*ms[d-1], aug_rs[d], grid2);
    }

    SUPERLU_FREE(aug_rs);
}

void dconvertTT_tensor(double **TTcores_global, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, double *X, int *grid_proc)
{
    int_t i, j, k;
    double **global_TTcores_root, **tmp;
    int *TTcore_col, *reconstruct_row, *aug_rs;
    double one = 1.0, zero = 0.0;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    if ( !(TTcore_col = intMalloc_dist(d)) )
        ABORT("Malloc fails for TTcore_col[]");
    for (k = 0; k < d; ++k) {
        TTcore_col[k] = ms[k]*aug_rs[k+1];
    }

    if ( !(reconstruct_row = intMalloc_dist(d-1)) )
        ABORT("Malloc fails for reconstruct_row[]");
    reconstruct_row[0] = ms[0];
    for (k = 1; k < d-1; ++k) {
        reconstruct_row[k] = reconstruct_row[k-1] * ms[k];
    }

    if (grids[0]->iam == 0) {
        global_TTcores_root = (double **) SUPERLU_MALLOC(d*sizeof(double*));
        for (k = 0; k < d; ++k) {
            if ( !(global_TTcores_root[k] = doubleMalloc_dist(aug_rs[k]*ms[k]*aug_rs[k+1])) )
                ABORT("Malloc fails for global_TTcores_root[k][]");
        }
        for (j = 0; j < rs[0]; ++j) {
            for (i = 0; i < ms[0]; ++i) {
                global_TTcores_root[0][j*ms[0]+i] = TTcores_global[0][j*ms[0]+i];
            }
        }
    }

    for (k = 1; k < d; ++k) {
        transfer_X_dgrids(TTcores_global[k], aug_rs[k]*ms[k], aug_rs[k+1], global_TTcores_root[k], grid_proc, k, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (grids[0]->iam == 0) {
        tmp = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));
        if ( !(tmp[0] = doubleMalloc_dist(reconstruct_row[0]*TTcore_col[1])) )
            ABORT("Malloc fails for tmp[0][]");
        dgemm_("N", "N", &(reconstruct_row[0]), &(TTcore_col[1]), &(aug_rs[1]), &one, global_TTcores_root[0], &(reconstruct_row[0]), 
            global_TTcores_root[1], &(aug_rs[1]), &zero, tmp[0], &(reconstruct_row[0]));

        for (k = 1; k < d-2; ++k) {
            if ( !(tmp[k] = doubleMalloc_dist(reconstruct_row[k]*TTcore_col[k+1])) )
                ABORT("Malloc fails for tmp[k][]");
            dgemm_("N", "N", &(reconstruct_row[k]), &(TTcore_col[k+1]), &(aug_rs[k+1]), &one, tmp[k-1], &(reconstruct_row[k]), 
                global_TTcores_root[k+1], &(aug_rs[k+1]), &zero, tmp[k], &(reconstruct_row[k]));
        }
        dgemm_("N", "N", &(reconstruct_row[d-2]), &(TTcore_col[d-1]), &(aug_rs[d-1]), &one, tmp[d-3], &(reconstruct_row[d-2]), 
            global_TTcores_root[d-1], &(aug_rs[d-1]), &zero, X, &(reconstruct_row[d-2]));

        for (k = 0; k < d-2; ++k) {
            SUPERLU_FREE(tmp[k]);
        }
        SUPERLU_FREE(tmp);
        for (k = 0; k < d; ++k) {
            SUPERLU_FREE(global_TTcores_root[k]);
        }
        SUPERLU_FREE(global_TTcores_root);
    }

    SUPERLU_FREE(TTcore_col);
    SUPERLU_FREE(reconstruct_row);
    SUPERLU_FREE(aug_rs);
}

void dconvertTT_tensor_2grids(double **TTcores_global, gridinfo_t *grid1, gridinfo_t *grid2, int *ms, int *rs, int *locals, 
    int d, double *X, int *grid_proc)
{
    int_t i, j, k;
    double **tmp;
    double *TTcore_last;
    int *TTcore_col, *reconstruct_row, *aug_rs;
    double one = 1.0, zero = 0.0;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    if ( !(TTcore_col = intMalloc_dist(d)) )
        ABORT("Malloc fails for TTcore_col[]");
    for (k = 0; k < d; ++k) {
        TTcore_col[k] = ms[k]*aug_rs[k+1];
    }

    if ( !(reconstruct_row = intMalloc_dist(d-1)) )
        ABORT("Malloc fails for reconstruct_row[]");
    reconstruct_row[0] = ms[0];
    for (k = 1; k < d-1; ++k) {
        reconstruct_row[k] = reconstruct_row[k-1] * ms[k];
    }

    if (grid1->iam == 0) {
        if ( !(TTcore_last = doubleMalloc_dist(rs[d-2]*ms[d-1])) )
            ABORT("Malloc fails for TTcore_last[]");
    }
    transfer_X_dgrids(TTcores_global[d-1], rs[d-2], ms[d-1], TTcore_last, grid_proc, 1, 0);

    if (grid1->iam == 0) {
        tmp = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));
        if ( !(tmp[0] = doubleMalloc_dist(reconstruct_row[0]*TTcore_col[1])) )
            ABORT("Malloc fails for tmp[0][]");
        dgemm_("N", "N", &(reconstruct_row[0]), &(TTcore_col[1]), &(aug_rs[1]), &one, TTcores_global[0], &(reconstruct_row[0]), 
            TTcores_global[1], &(aug_rs[1]), &zero, tmp[0], &(reconstruct_row[0]));

        for (k = 1; k < d-2; ++k) {
            if ( !(tmp[k] = doubleMalloc_dist(reconstruct_row[k]*TTcore_col[k+1])) )
                ABORT("Malloc fails for tmp[k][]");
            dgemm_("N", "N", &(reconstruct_row[k]), &(TTcore_col[k+1]), &(aug_rs[k+1]), &one, tmp[k-1], &(reconstruct_row[k]), 
                TTcores_global[k+1], &(aug_rs[k+1]), &zero, tmp[k], &(reconstruct_row[k]));
        }
        dgemm_("N", "N", &(reconstruct_row[d-2]), &(TTcore_col[d-1]), &(aug_rs[d-1]), &one, tmp[d-3], &(reconstruct_row[d-2]), 
            TTcore_last, &(aug_rs[d-1]), &zero, X, &(reconstruct_row[d-2]));

        for (k = 0; k < d-2; ++k) {
            SUPERLU_FREE(tmp[k]);
        }
        SUPERLU_FREE(tmp);
        
        SUPERLU_FREE(TTcore_last);
    }

    SUPERLU_FREE(TTcore_col);
    SUPERLU_FREE(reconstruct_row);
    SUPERLU_FREE(aug_rs);
}

void dcheck_error_TT(int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs, gridinfo_t **grids,
    int *rs, int *locals, int d, double *F, double **TTcores, double *trueX, int *grid_proc)
{
    int *aug_rs;
    char     transpose[1];
    *transpose = 'N';
    double *X, *AX, *RHS;
    double **TTcores_global;
    int_t i, j, k, l, t;
    double err1 = 0.0, err2 = 0.0, norm1 = 0.0, norm2 = 0.0;
    double s;
    double one = 1.0, zero = 0.0;
    int nelem = 1;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    for (k = 0; k < d; ++k) {
        nelem = nelem * ms[k];
    }
    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    // printf("proc %d knows rank %d, %d, %d, %d.\n", global_rank, aug_rs[0], aug_rs[1], aug_rs[2], aug_rs[3]);
    // fflush(stdout);

    TTcores_global = (double **) SUPERLU_MALLOC(d*sizeof(double*));
    dgather_TTcores(TTcores, grids, ms, rs, locals, d, TTcores_global);

    if (grids[0]->iam == 0) {
        if ( !(X = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for X[]");
        if ( !(AX = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for AX[]");
        if ( !(RHS = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for RHS[]");

        for (k = 0; k < nelem; ++k) {
            RHS[k] = 0.0;
        }
    }
    dconvertTT_tensor(TTcores_global, grids, ms, rs, locals, d, X, grid_proc);

    // if (grids[0]->iam == 0) {
    //     printf("Get reconstructed solution X from TT cores.\n");
    //     for (j = 0; j < nelem; ++j) {
    //         printf("%f ", X[j]);
    //     }
    //     printf("\n");
    //     printf("True X is.\n");
    //     for (j = 0; j < nelem; ++j) {
    //         printf("%f ", trueX[j]);
    //     }
    //     printf("\n");
    //     fflush(stdout);
    // }

    for (t = 0; t < d; ++t) {
        double **TTcores_update = (double **) SUPERLU_MALLOC(d*sizeof(double*));
        for (l = 0; l < d; ++l) {
            if (grids[l]->iam == 0) {
                if ( !(TTcores_update[l] = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                    ABORT("Malloc fails for TTcores_update[l][]");

                if (l == t) {
                    SuperMatrix GA;
                    /* Create compressed column matrix for GA. */
                    // printf("ms is %d, nnz is %d.\n", ms[l], nnzs[l]);
                    dCreate_CompCol_Matrix_dist(&GA, ms[l], ms[l], nnzs[l], nzvals[l], rowinds[l], colptrs[l],
                        SLU_NC, SLU_D, SLU_GE);

                    double *tmp1, *tmp2;
                    if ( !(tmp1 = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                        ABORT("Malloc fails for tmp1[]");
                    for (k = 0; k < aug_rs[l+1]; ++k) {
                        for (j = 0; j < aug_rs[l]; ++j) {
                            for (i = 0; i < ms[l]; ++i) {
                                tmp1[k*aug_rs[l]*ms[l]+j*ms[l]+i] = TTcores_global[l][k*aug_rs[l]*ms[l]+i*aug_rs[l]+j];
                            }
                        }
                    }
                    // printf("Rearrange succeeded.\n");
                    // for (i = 0; i < ms[l]; ++i) {
                    //     for (j = 0; j < aug_rs[l]*aug_rs[l+1]; ++j) {
                    //         printf("%f ", tmp1[j*ms[l]+i]);
                    //     }
                    //     printf("\n");
                    // }
                    // fflush(stdout);
                    if ( !(tmp2 = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                        ABORT("Malloc fails for tmp2[]");
                    sp_dgemm_dist(transpose, aug_rs[l]*aug_rs[l+1], 1.0, &GA, tmp1, ms[l], 0.0, tmp2, ms[l]);
                    // printf("Multiplication succeeded.\n");
                    // fflush(stdout);
                    for (k = 0; k < aug_rs[l+1]; ++k) {
                        for (j = 0; j < ms[l]; ++j) {
                            for (i = 0; i < aug_rs[l]; ++i) {
                                TTcores_update[l][k*aug_rs[l]*ms[l]+j*aug_rs[l]+i] = tmp2[k*aug_rs[l]*ms[l]+i*ms[l]+j];
                            }
                        }
                    }

                    SUPERLU_FREE(tmp1);
                    SUPERLU_FREE(tmp2);

                    /* Destroy GA */
                    Destroy_CompCol_Matrix_dist(&GA);
                }
                else {
                    for (j = 0; j < aug_rs[l]*ms[l]*aug_rs[l+1]; ++j) {
                        TTcores_update[l][j] = TTcores_global[l][j];
                    }
                }
            }
        }
        // printf("proc %d is here for reconstruction %d.\n", global_rank, t);
        // fflush(stdout);
        dconvertTT_tensor(TTcores_update, grids, ms, rs, locals, d, AX, grid_proc);
        
        if (grids[0]->iam == 0) {
            for (j = 0; j < nelem; ++j) {
                RHS[j] += AX[j];
            }
        }

        for (l = 0; l < d; ++l) {
            if (grids[l]->iam == 0) {
                SUPERLU_FREE(TTcores_update[l]);
            }
        }
        SUPERLU_FREE(TTcores_update);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // if (grids[0]->iam == 0) {
    //     printf("Get reconstructed RHS.\n");
    //     for (j = 0; j < 10; ++j) {
    //         printf("%f ", RHS[j]);
    //     }
    //     printf("\n");
    //     printf("True RHS is.\n");
    //     for (j = 0; j < 10; ++j) {
    //         printf("%f ", F[j]);
    //     }
    //     printf("\n");
    //     fflush(stdout);
    // }

    if (grids[0]->iam == 0) {
        for (j = 0; j < nelem; ++j) {
            s = F[j] - RHS[j];
            err1 += s*s;

            s = X[j] - trueX[j];
            err2 += s*s;

            s = F[j];
            norm1 += s*s;

            s = trueX[j];
            norm2 += s*s;
        }
        
        err1 = sqrt(err1) / sqrt(norm1);
        err2 = sqrt(err2) / sqrt(norm2);
        printf("Relative error of approximating RHS is %f, and approximating true solution is %f.\n", err1, err2);
        fflush(stdout);

        SUPERLU_FREE(X);
        SUPERLU_FREE(AX);
        SUPERLU_FREE(RHS);
    }

    for (l = 0; l < d; ++l) {
        if (grids[l]->iam == 0) {
            SUPERLU_FREE(TTcores_global[l]);
        }
    }
    SUPERLU_FREE(TTcores_global);
    SUPERLU_FREE(aug_rs);
}

void dcheck_error_TT_2grids(int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs, gridinfo_t *grid1, gridinfo_t *grid2,
    int *rs, int *locals, int d, double *F, double **TTcores, double *trueX, int *grid_proc)
{
    int *aug_rs;
    char transpose[1];
    *transpose = 'N';
    double *X, *AX, *RHS;
    double **TTcores_global;
    int_t i, j, k, l, t;
    double err1 = 0.0, err2 = 0.0, norm1 = 0.0, norm2 = 0.0;
    double s;
    double one = 1.0, zero = 0.0;
    int nelem = 1;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    for (k = 0; k < d; ++k) {
        nelem = nelem * ms[k];
    }
    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    // printf("proc %d knows rank", global_rank);
    // for (j = 0; j < d+1; ++j) {
    //     printf(" %d", aug_rs[j]);
    // }
    // printf("\n");
    // fflush(stdout);

    TTcores_global = (double **) SUPERLU_MALLOC(d*sizeof(double*));
    dgather_TTcores_2grids(TTcores, grid1, grid2, ms, rs, locals, d, TTcores_global);

    if (grid1->iam == 0) {
        if ( !(X = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for X[]");
        if ( !(AX = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for AX[]");
        if ( !(RHS = doubleMalloc_dist(nelem)) )
            ABORT("Malloc fails for RHS[]");

        for (k = 0; k < nelem; ++k) {
            RHS[k] = 0.0;
        }
    }
    dconvertTT_tensor_2grids(TTcores_global, grid1, grid2, ms, rs, locals, d, X, grid_proc);

    // if (grid1->iam == 0) {
    //     printf("Get reconstructed solution X from TT cores.\n");
    //     for (j = 0; j < nelem; ++j) {
    //         printf("%f ", X[j]);
    //     }
    //     printf("\n");
    //     printf("True X is.\n");
    //     for (j = 0; j < nelem; ++j) {
    //         printf("%f ", trueX[j]);
    //     }
    //     printf("\n");
    //     fflush(stdout);
    // }

    for (t = 0; t < d; ++t) {
        double **TTcores_update = (double **) SUPERLU_MALLOC(d*sizeof(double*));
        if (grid1->iam == 0) {
            for (l = 0; l < d-1; ++l) {
                if ( !(TTcores_update[l] = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                    ABORT("Malloc fails for TTcores_update[l][]");

                if (l == t) {
                    SuperMatrix GA;
                    dCreate_CompCol_Matrix_dist(&GA, ms[l], ms[l], nnzs[l], nzvals[l], rowinds[l], colptrs[l],
                        SLU_NC, SLU_D, SLU_GE);

                    double *tmp1, *tmp2;
                    if ( !(tmp1 = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                        ABORT("Malloc fails for tmp1[]");
                    for (k = 0; k < aug_rs[l+1]; ++k) {
                        for (j = 0; j < aug_rs[l]; ++j) {
                            for (i = 0; i < ms[l]; ++i) {
                                tmp1[k*aug_rs[l]*ms[l]+j*ms[l]+i] = TTcores_global[l][k*aug_rs[l]*ms[l]+i*aug_rs[l]+j];
                            }
                        }
                    }

                    // printf("Done reordering on grid 1 for iter %d for l and %d for t.\n", l, t);
                    // for (i = 0; i < ms[l]; ++i) {
                    //     for (j = 0; j < aug_rs[l]*aug_rs[l+1]; ++j) {
                    //         printf("%f ", tmp1[j*ms[l]+i]);
                    //     }
                    //     printf("\n");
                    // }
                    // fflush(stdout);
                    
                    if ( !(tmp2 = doubleMalloc_dist(aug_rs[l]*ms[l]*aug_rs[l+1])) )
                        ABORT("Malloc fails for tmp2[]");
                    sp_dgemm_dist(transpose, aug_rs[l]*aug_rs[l+1], 1.0, &GA, tmp1, ms[l], 0.0, tmp2, ms[l]);

                    // printf("Done multiplication on grid 1 for iter %d for l and %d for t.\n", l, t);
                    // for (i = 0; i < ms[l]; ++i) {
                    //     for (j = 0; j < aug_rs[l]*aug_rs[l+1]; ++j) {
                    //         printf("%f ", tmp2[j*ms[l]+i]);
                    //     }
                    //     printf("\n");
                    // }
                    // fflush(stdout);
                    
                    for (k = 0; k < aug_rs[l+1]; ++k) {
                        for (j = 0; j < ms[l]; ++j) {
                            for (i = 0; i < aug_rs[l]; ++i) {
                                TTcores_update[l][k*aug_rs[l]*ms[l]+j*aug_rs[l]+i] = tmp2[k*aug_rs[l]*ms[l]+i*ms[l]+j];
                            }
                        }
                    }

                    // printf("Done reorganization on grid 1 for iter %d for l and %d for t.\n", l, t);
                    // fflush(stdout);

                    SUPERLU_FREE(tmp1);
                    SUPERLU_FREE(tmp2);

                    /* Destroy GA */
                    Destroy_CompCol_Matrix_dist(&GA);
                }
                else {
                    for (j = 0; j < aug_rs[l]*ms[l]*aug_rs[l+1]; ++j) {
                        TTcores_update[l][j] = TTcores_global[l][j];
                    }
                }
                // printf("Done mult on grid 1 for iter %d.\n", l);
                // fflush(stdout);
            }
        }
        else if (grid2->iam == 0) {
            if ( !(TTcores_update[d-1] = doubleMalloc_dist(aug_rs[d-1]*ms[d-1]*aug_rs[d])) )
                ABORT("Malloc fails for TTcores_update[d-1][]");

            if (t == d-1) {
                SuperMatrix GA;
                dCreate_CompCol_Matrix_dist(&GA, ms[d-1], ms[d-1], nnzs[d-1], nzvals[d-1], rowinds[d-1], colptrs[d-1],
                    SLU_NC, SLU_D, SLU_GE);

                double *tmp1, *tmp2;
                if ( !(tmp1 = doubleMalloc_dist(aug_rs[d-1]*ms[d-1]*aug_rs[d])) )
                    ABORT("Malloc fails for tmp1[]");
                for (k = 0; k < aug_rs[d]; ++k) {
                    for (j = 0; j < aug_rs[d-1]; ++j) {
                        for (i = 0; i < ms[d-1]; ++i) {
                            tmp1[k*aug_rs[d-1]*ms[d-1]+j*ms[d-1]+i] = TTcores_global[d-1][k*aug_rs[d-1]*ms[d-1]+i*aug_rs[d-1]+j];
                        }
                    }
                }
                
                if ( !(tmp2 = doubleMalloc_dist(aug_rs[d-1]*ms[d-1]*aug_rs[d])) )
                    ABORT("Malloc fails for tmp2[]");
                sp_dgemm_dist(transpose, aug_rs[d-1]*aug_rs[d], 1.0, &GA, tmp1, ms[d-1], 0.0, tmp2, ms[d-1]);
                
                for (k = 0; k < aug_rs[d]; ++k) {
                    for (j = 0; j < ms[d-1]; ++j) {
                        for (i = 0; i < aug_rs[d-1]; ++i) {
                            TTcores_update[d-1][k*aug_rs[d-1]*ms[d-1]+j*aug_rs[d-1]+i] = tmp2[k*aug_rs[d-1]*ms[d-1]+i*ms[d-1]+j];
                        }
                    }
                }

                SUPERLU_FREE(tmp1);
                SUPERLU_FREE(tmp2);

                /* Destroy GA */
                Destroy_CompCol_Matrix_dist(&GA);
            }
            else {
                for (j = 0; j < aug_rs[d-1]*ms[d-1]*aug_rs[d]; ++j) {
                    TTcores_update[d-1][j] = TTcores_global[d-1][j];
                }
            }
        }
        // printf("proc %d is here for reconstruction %d.\n", global_rank, t);
        // fflush(stdout);
        dconvertTT_tensor_2grids(TTcores_update, grid1, grid2, ms, rs, locals, d, AX, grid_proc);
        
        if (grid1->iam == 0) {
            for (j = 0; j < nelem; ++j) {
                RHS[j] += AX[j];
            }

            for (l = 0; l < d-1; ++l) {
                SUPERLU_FREE(TTcores_update[l]);
            }
        }
        if (grid2->iam == 0) {
            SUPERLU_FREE(TTcores_update[d-1]);
        }
        SUPERLU_FREE(TTcores_update);

        if (grid1->iam != -1) {
            MPI_Barrier(grid1->comm);
        }
        if (grid2->iam != -1) {
            MPI_Barrier(grid2->comm);
        }
    }
    // printf("proc %d is here.\n", global_rank);
    // fflush(stdout);

    if (grid1->iam == 0) {
        for (j = 0; j < nelem; ++j) {
            s = F[j] - RHS[j];
            err1 += s*s;

            s = X[j] - trueX[j];
            err2 += s*s;

            s = F[j];
            norm1 += s*s;

            s = trueX[j];
            norm2 += s*s;
        }
        
        err1 = sqrt(err1) / sqrt(norm1);
        err2 = sqrt(err2) / sqrt(norm2);
        printf("Relative error of approximating RHS is %.10e, and approximating true solution is %.10e.\n", err1, err2);
        fflush(stdout);

        SUPERLU_FREE(X);
        SUPERLU_FREE(AX);
        SUPERLU_FREE(RHS);
    }

    if (grid1->iam == 0) {
        for (l = 0; l < d-1; ++l) {
            SUPERLU_FREE(TTcores_global[l]);
        }
    }
    if (grid2->iam == 0) {
        SUPERLU_FREE(TTcores_global[d-1]);
    }
    SUPERLU_FREE(TTcores_global);
    SUPERLU_FREE(aug_rs);
}

void dredistribute_X_twogrids(double *X, double *F, double *F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t m_A, int_t m_B, int_t r, int *grid_proc, int send_grid, int recv_grid)
{
    int_t i, j, k;
    int_t m_loc, m_loc_fst, row, fst_row;
    int iam, grA = 0, grB = 0;
    int nproc_A = grid_A->nprow * grid_A->npcol;
    int nproc_B = grid_B->nprow * grid_B->npcol;
    double *rX;
    
    if (grid_A->iam == 0) {
        MPI_Bcast(X, m_A*m_B*r, MPI_DOUBLE, 0, grid_A->comm);
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(X, m_A*m_B*r, MPI_DOUBLE, 0, grid_A->comm);
    }

    if (grid_B->iam != -1) {
        rX = (double *) doubleMalloc_dist(m_A*m_B*r);
    }

    transfer_X_dgrids(X, m_A*m_B, r, rX, grid_proc, send_grid, recv_grid);

    if (grid_B->iam != -1) {
        MPI_Bcast(rX, m_A*m_B*r, MPI_DOUBLE, 0, grid_B->comm);
    }

    if (grid_A->iam != -1) {
        m_loc = m_A / nproc_A; 
        m_loc_fst = m_loc;
        iam = grid_A->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_A) != m_A) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_A - 1)) /* last proc. gets all*/
          m_loc = m_A - m_loc * (nproc_A - 1);
        }

        /* Get the local B */
        for (k = 0; k < r; ++k) {
            for (j = 0; j < m_B; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;
                    F[k*m_loc*m_B+j*m_loc+i] = X[k*m_A*m_B+j*m_A+row];
                }
            }
        }
    }

    if (grid_B->iam != -1) {
        m_loc = m_B / nproc_B; 
        m_loc_fst = m_loc;
        iam = grid_B->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_B) != m_B) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_B - 1)) /* last proc. gets all*/
          m_loc = m_B - m_loc * (nproc_B - 1);
        }

        /* Get the local B */
        for (k = 0; k < r; ++k) {
            for (j = 0; j < m_A; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;
                    F_transpose[k*m_loc*m_A+j*m_loc+i] = rX[k*m_A*m_B+row*m_A+j];
                }
            }
        }
        SUPERLU_FREE(rX);
    }
}


void dTT_right_orthonormalization(double **TTcores, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, int *grid_proc)
{
    int_t i, j, k, l;
    int *aug_rs, *TTcore_row;
    double *localX, *localQ, *R, *rR;
    double one = 1.0, zero = 0.0;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    if ( !(TTcore_row = intMalloc_dist(d)) )
        ABORT("Malloc fails for TTcore_row[]");
    for (k = 0; k < d; ++k) {
        TTcore_row[k] = aug_rs[k]*locals[k];
    }

    for (l = 0; l < d; ++l) {
        if (grids[l]->iam != -1) {
            if ( !(localX = doubleMalloc_dist(locals[l]*aug_rs[l+1]*aug_rs[l])) )
                ABORT("Malloc fails for localX[]");
        }
    }

    for (l = d-1; l > 0; --l) {
        if (grids[l]->iam != -1) {
            for (j = 0; j < aug_rs[l]; ++j) {
                for (i = 0; i < locals[l]*aug_rs[l+1]; ++i) {
                    localX[j*locals[l]*aug_rs[l+1]+i] = TTcores[l][i*aug_rs[l]+j];
                }
            }

            dQR_dist(localX, locals[l]*aug_rs[l+1], &localQ, &R, aug_rs[l], grids[l]);
            for (j = 0; j < locals[l]*aug_rs[l+1]; ++j) {
                for (i = 0; i < aug_rs[l]; ++i) {
                    TTcores[l][j*aug_rs[l]+i] = localQ[i*locals[l]*aug_rs[l+1]+j];
                }
            }

            SUPERLU_FREE(localQ);

            MPI_Barrier(grids[l]->comm);
        }

        if (grids[l-1]->iam != -1) {
            if ( !(rR = doubleMalloc_dist(aug_rs[l]*aug_rs[l])) )
                ABORT("Malloc fails for rR[]");
        }
        transfer_X_dgrids(R, aug_rs[l], aug_rs[l], rR, grid_proc, l, l-1);

        if (grids[l-1]->iam != -1) {
            MPI_Bcast(rR, aug_rs[l]*aug_rs[l], MPI_DOUBLE, 0, grids[l-1]->comm);

            dgemm_("N", "N", &(TTcore_row[l-1]), &(aug_rs[l]), &(aug_rs[l]), &one, TTcores[l-1], &(TTcore_row[l-1]), 
                rR, &(aug_rs[l]), &zero, localX, &(TTcore_row[l-1]));

            for (j = 0; j < aug_rs[l]; ++j) {
                for (i = 0; i < locals[l-1]*aug_rs[l-1]; ++i) {
                    TTcores[l-1][j*locals[l-1]*aug_rs[l-1]+i] = localX[j*locals[l-1]*aug_rs[l-1]+i];
                }
            }
        }

        if (grids[l]->iam == 0) {
            SUPERLU_FREE(R);
        }
        if (grids[l-1]->iam != -1) {
            SUPERLU_FREE(rR);
        }
    }

    for (l = 0; l < d; ++l) {
        if (grids[l]->iam != -1) {
            SUPERLU_FREE(localX);
        }
    }
    SUPERLU_FREE(TTcore_row);
    SUPERLU_FREE(aug_rs);
}

void dTT_rounding(double **TTcores, double **TTcores_new, gridinfo_t **grids, int *ms, int *rs, int *locals, int d, int *grid_proc, double tol)
{
    int_t i, j, k, l;
    int *aug_rs, *TTcore_row, *TTcore_col;
    double *localX, *localQ, *R, *localU, *localS, *localV, *rV;
    double one = 1.0, zero = 0.0, newtol = 0.0, fro_norm = 0.0;
    int newr, gridroot;

    if ( !(aug_rs = intMalloc_dist(d+1)) )
        ABORT("Malloc fails for aug_rs[]");
    aug_rs[0] = 1;
    for (k = 0; k < d-1; ++k) {
        aug_rs[k+1] = rs[k];
    }
    aug_rs[d] = 1;

    if ( !(TTcore_row = intMalloc_dist(d)) )
        ABORT("Malloc fails for TTcore_row[]");
    if ( !(TTcore_col = intMalloc_dist(d)) )
        ABORT("Malloc fails for TTcore_col[]");
    for (k = 0; k < d; ++k) {
        TTcore_row[k] = aug_rs[k]*locals[k];
        TTcore_col[k] = aug_rs[k+1]*locals[k];
    }

    dTT_right_orthonormalization(TTcores, grids, ms, rs, locals, d, grid_proc);

    if (grids[0]->iam != -1) {
        if ( !(TTcores_new[0] = doubleMalloc_dist(locals[0]*rs[0])) )
            ABORT("Malloc fails for TTcores_new[0][]");

        for (j = 0; j < rs[0]; ++j) {
            for (i = 0; i < locals[0]; ++i) {
                fro_norm += TTcores[0][j*locals[0]+i] * TTcores[0][j*locals[0]+i];

                TTcores_new[0][j*locals[0]+i] = TTcores[0][j*locals[0]+i];
            }
        }
        MPI_Reduce(&fro_norm, &newtol, 1, MPI_DOUBLE, MPI_SUM, 0, grids[0]->iam);
    }
    MPI_Bcast(&newtol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    newtol = sqrt(newtol) / (d-1) * tol;

    for (l = 0; l < d-1; ++l) {
        gridroot = 0;
        for (k = 0; k < l; ++k) {
            gridroot += grid_proc[k];
        }

        if (grids[l]->iam != -1) {
            dQR_dist(TTcores_new[l], TTcore_row[l], &localQ, &R, rs[l], grids[l]);

            if (grids[l]->iam == 0) {
                dtruncated_SVD(R, rs[l], rs[l], &newr, &localU, &localS, &localV, newtol);
                for (j = 0; j < newr; ++j) {
                    for (i = 0; i < rs[l]; ++i) {
                        localV[j*rs[l]+i] = localV[j*rs[l]+i]*localS[j];
                    }
                }
                SUPERLU_FREE(localS);
            }
        }

        MPI_Bcast(&newr, 1, MPI_INT, gridroot, MPI_COMM_WORLD);

        if (grids[l]->iam != -1) {
            if (grids[l]->iam != 0) {
                if ( !(localU = doubleMalloc_dist(rs[l]*newr)) )
                    ABORT("Malloc fails for localU[]");
            }
            MPI_Bcast(&localU, rs[l]*newr, MPI_DOUBLE, 0, grids[l]->comm);
            dgemm_("N", "N", &(TTcore_row[l]), &newr, &(rs[l]), &one, localQ, &(TTcore_row[l]), 
                localU, &(rs[l]), &zero, TTcores_new[l], &(TTcore_row[l]));

            SUPERLU_FREE(localQ);
            SUPERLU_FREE(localU);
        }

        if (grids[l+1]->iam != -1) {
            if ( !(rV = doubleMalloc_dist(rs[l]*newr)) )
                ABORT("Malloc fails for rV[]");
        }
        transfer_X_dgrids(localV, rs[l], newr, rV, grid_proc, l, l+1);

        if (grids[l]->iam == 0) {
            SUPERLU_FREE(localV);
        }

        if (grids[l+1]->iam != -1) {
            MPI_Bcast(rV, rs[l]*newr, MPI_DOUBLE, 0, grids[l+1]->comm);

            if ( !(TTcores_new[l+1] = doubleMalloc_dist(rs[l]*newr)) )
                ABORT("Malloc fails for TTcores_new[l+1][]");

            dgemm_("T", "N", &(newr), &(TTcore_col[l+1]), &(rs[l]), &one, rV, &(rs[l]), 
                TTcores[l+1], &(rs[l]), &zero, TTcores_new[l+1], &(newr));

            SUPERLU_FREE(rV);
        }

        rs[l] = newr;
    }

    SUPERLU_FREE(TTcore_row);
    SUPERLU_FREE(TTcore_col);
    SUPERLU_FREE(aug_rs);
}

void dmult_TTfADI_mat(int_t m_A, double *A, int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    double *U, int r, double *X)
{
    SuperMatrix GB;
    int_t i, j, k;
    int tcol, trow;
    double *T1, *T2, *T3;
    double one = 1.0, zero = 0.0;
    char transpose[1];
    *transpose = 'N';

    if ( !(T1 = doubleMalloc_dist(m_A*m_B*r)) )
        ABORT("Malloc fails for T1[]");
    tcol = m_B*r;
    dgemm_("N", "N", &m_A, &tcol, &m_A, &one, A, &m_A, U, &m_A, &zero, T1, &m_A);

    if ( !(T2 = doubleMalloc_dist(m_A*m_B*r)) )
        ABORT("Malloc fails for T2[]");
    for (k = 0; k < r; ++k) {
        for (j = 0; j < m_A; ++j) {
            for (i = 0; i < m_B; ++i) {
                T2[k*m_A*m_B+j*m_B+i] = U[k*m_A*m_B+i*m_A+j];
            }
        }
    }

    if ( !(T3 = doubleMalloc_dist(m_A*m_B*r)) )
        ABORT("Malloc fails for T3[]");
    dCreate_CompCol_Matrix_dist(&GB, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        SLU_NC, SLU_D, SLU_GE);
    sp_dgemm_dist(transpose, r, one, &GB, T2, m_B, zero, T3, m_B);
    Destroy_CompCol_Matrix_dist(&GB);

    for (k = 0; k < r; ++k) {
        for (j = 0; j < m_B; ++j) {
            for (i = 0; i < m_A; ++i) {
                T1[k*m_A*m_B+j*m_A+i] += T3[k*m_A*m_B+i*m_B+j];
            }
        }
    }

    trow = m_A*m_B;
    dgemm_("T", "N", &r, &r, &trow, &one, U, &trow, T1, &trow, &zero, X, &r);

    SUPERLU_FREE(T1);
    SUPERLU_FREE(T2);
    SUPERLU_FREE(T3);
}

void dmult_TTfADI_RHS(int_t *ms, int_t *rs, int_t local, int ddeal, double *M, int nrhs, double **TTcores_global, double **newM)
{
    int rowM = 1, rowtmp, rowtmp_mid, coltmp;
    int_t k;
    double **tmp_mat;
    double one = 1.0, zero = 0.0;

    for (k = 0; k < ddeal; ++k) {
        rowM *= ms[k];
    }

    rowtmp = rs[0];
    rowtmp_mid = ms[0];
    rowM = rowM / ms[0];
    coltmp = rowM * local * nrhs;

    if (ddeal == 1) {
        if ( !(*newM = doubleMalloc_dist(rowtmp*coltmp)) )
            ABORT("Malloc fails for *newM[]");
        dgemm_("T", "N", &rowtmp, &coltmp, &rowtmp_mid, &one, TTcores_global[0], &rowtmp_mid, M, &rowtmp_mid, &zero, *newM, &rowtmp);
        return;
    }

    tmp_mat = (double **) SUPERLU_MALLOC((ddeal-1)*sizeof(double*));
    if ( !(tmp_mat[0] = doubleMalloc_dist(rowtmp*coltmp)) )
        ABORT("Malloc fails for tmp_mat[0][]");
    dgemm_("T", "N", &rowtmp, &coltmp, &rowtmp_mid, &one, TTcores_global[0], &rowtmp_mid, M, &rowtmp_mid, &zero, tmp_mat[0], &rowtmp);

    for (k = 1; k < ddeal-1; ++k) {
        rowtmp = rs[k];
        rowtmp_mid = ms[k] * rs[k-1];
        rowM = rowM / ms[k];
        coltmp = rowM * local * nrhs;

        if ( !(tmp_mat[k] = doubleMalloc_dist(rowtmp*coltmp)) )
            ABORT("Malloc fails for tmp_mat[k][]");
        dgemm_("T", "N", &rowtmp, &coltmp, &rowtmp_mid, &one, TTcores_global[k], &rowtmp_mid, 
            tmp_mat[k-1], &rowtmp_mid, &zero, tmp_mat[k], &rowtmp);
    }

    rowtmp = rs[ddeal-1];
    rowtmp_mid = ms[ddeal-1] * rs[ddeal-2];
    coltmp = local * nrhs;
    if ( !(*newM = doubleMalloc_dist(rowtmp*coltmp)) )
        ABORT("Malloc fails for *newM[]");
    dgemm_("T", "N", &rowtmp, &coltmp, &rowtmp_mid, &one, TTcores_global[ddeal-1], &rowtmp_mid, 
        tmp_mat[ddeal-2], &rowtmp_mid, &zero, *newM, &rowtmp);

    for (k = 0; k < ddeal-1; ++k) {
        SUPERLU_FREE(tmp_mat[k]);
    }
    SUPERLU_FREE(tmp_mat);
}