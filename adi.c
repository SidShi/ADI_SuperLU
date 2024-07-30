#include <math.h>
#include "superlu_ddefs.h"
#include "adi.h"

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

void adi(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *F, int ldf, double *F_transpose, int ldft,
    double *p, double *q, int_t l, double *X)
{
    SuperMatrix A, B;
    double *rhs_A, *rhs_B;
    double *Y;
    int ldx, info;
    int_t k;
    double *berr_A, *berr_B;
    SuperLUStat_t stat_A, stat_B;
    dScalePermstruct_t ScalePermstruct_A, ScalePermstruct_B;
    dLUstruct_t LUstruct_A, LUstruct_B;
    dSOLVEstruct_t SOLVEstruct_A, SOLVEstruct_B;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    ldx = m_A;

    if (grid_A->iam != -1) {
        if ( !(berr_A = doubleMalloc_dist(m_B)) )
            ABORT("Malloc fails for berr_A[].");
    }
    if (grid_B->iam != -1) {
        if ( !(berr_B = doubleMalloc_dist(m_A)) )
            ABORT("Malloc fails for berr_B[].");
        if ( !(Y = doubleMalloc_dist(m_B*m_A)) )
            ABORT("Malloc fails for Y.");
    }

    for (k = 0; k < l; ++k) {
        if (grid_B->iam != -1) {
            dcreate_SuperMatrix(&B, grid_B, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, p[k]);
        }
        // printf("Global rank %d reaches here aa %d.\n", global_rank, k);
        // fflush(stdout);
        // MPI_Barrier(MPI_COMM_WORLD);

        dcreate_RHS(F_transpose, ldft, &rhs_B, 1, grid_A, grid_B, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, p[k], X, m_A, m_B);

        // printf("Global rank %d reaches here a %d.\n", global_rank, k);
        // fflush(stdout);
        // MPI_Barrier(MPI_COMM_WORLD);

        if (grid_B->iam != -1) {
            dScalePermstructInit(m_B, m_B, &ScalePermstruct_B);
            dLUstructInit(m_B, &LUstruct_B);
            PStatInit(&stat_B);
            pdgssvx(&options, &B, &ScalePermstruct_B, rhs_B, ldft, m_A, grid_B,
                &LUstruct_B, &SOLVEstruct_B, berr_B, &stat_B, &info);
            dgather_X(rhs_B, ldft, Y, m_B, m_A, grid_B);

            Destroy_CompRowLoc_Matrix_dist(&B);
            dScalePermstructFree(&ScalePermstruct_B);
            dDestroy_LU(m_B, grid_B, &LUstruct_B);
            dLUstructFree(&LUstruct_B);
            dSolveFinalize(&options, &SOLVEstruct_B);
        }

        // printf("Global rank %d reaches here b %d.\n", global_rank, k);
        // fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        if (grid_A->iam != -1) {
            dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[k]);
        }  
        dcreate_RHS(F, ldf, &rhs_A, 0, grid_A, grid_B, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, q[k], Y, m_B, m_A);

        // MPI_Barrier(MPI_COMM_WORLD);

        if (grid_A->iam != -1) {
            dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
            dLUstructInit(m_A, &LUstruct_A);
            PStatInit(&stat_A);
            pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldf, m_B, grid_A,
                &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);
            dgather_X(rhs_A, ldf, X, m_A, m_B, grid_A);

            Destroy_CompRowLoc_Matrix_dist(&A);
            dScalePermstructFree(&ScalePermstruct_A);
            dDestroy_LU(m_A, grid_A, &LUstruct_A);
            dLUstructFree(&LUstruct_A);
            dSolveFinalize(&options, &SOLVEstruct_A);
        }

        // printf("Global rank %d reaches here d %d.\n", global_rank, k);
        // fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if ( grid_A->iam != -1 ) {
        PStatFree(&stat_A);
        SUPERLU_FREE(berr_A);
        SUPERLU_FREE(rhs_A);
    }
    if ( grid_B->iam != -1 ) {
        PStatFree(&stat_B);
        SUPERLU_FREE(berr_B);
        SUPERLU_FREE(Y);
        SUPERLU_FREE(rhs_B);
    }
}

void adi_ls(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid, double *F, int local_b, int nrhs, double s, double la, double ua, double lb, double ub, double *X)
{
    SuperMatrix B;
    double *rhs_A, *rhs_B, *newA;
    int *ipiv;
    double *Y, *combined_localY;
    double *p, *q;
    double alpha, beta;
    int_t l;
    int info;
    int_t i, j, k, w;
    int_t n_solveA = local_b*nrhs;
    double *berr_B;
    SuperLUStat_t stat_B;
    dScalePermstruct_t ScalePermstruct_B;
    dLUstruct_t LUstruct_B;
    dSOLVEstruct_t SOLVEstruct_B;

    if ( !(berr_B = doubleMalloc_dist(m_A*nrhs)) )
        ABORT("Malloc fails for berr_B[].");
    if ( !(Y = doubleMalloc_dist(m_B*m_A*nrhs)) )
        ABORT("Malloc fails for Y.");
    if ( !(combined_localY = doubleMalloc_dist(local_b*nrhs*m_A)) )
        ABORT("Malloc fails for combined_localY.");
    if ( !(newA = doubleMalloc_dist(m_A*m_A)) )
        ABORT("Malloc fails for newA.");
    if ( !(ipiv = intMalloc_dist(m_A)) )
        ABORT("Malloc fails for ipiv.");

    for (j = 0; j < m_A; ++j) {
        for (i = 0; i < local_b*nrhs; ++i) {
            combined_localY[j*local_b*nrhs+i] = 0.0;
        }
    }

    alpha = s/2.0;
    beta = s/2.0;
    if (grid->iam == 0) {
        if (ua < lb)  {
            if (ua >= lb + s) {
                printf("ua is %f, lb is %f, s is %f.\n", ua, lb, s);
                fflush(stdout);
                ABORT("Inner ADI not ideal for solving.");
            }
        }
        else if (ub < la) {
            if (ub >= la - s) {
                printf("ub is %f, la is %f, s is %f.\n", ub, la, s);
                fflush(stdout);
                ABORT("Inner ADI not ideal for solving.");
            }
        }
    }

    dgenerate_shifts_onegrid(la-alpha, ua-alpha, lb+beta, ub+beta, &p, &q, &l, grid);

    // printf("Proc %d gets shifts of length %d with first elements %f and %f.\n", grid->iam, l, p[0], q[0]);
    // fflush(stdout);

    for (k = 0; k < l; ++k) {
        dcreate_SuperMatrix(&B, grid, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, -beta+p[k]);
        dcreate_RHS_ls(F, local_b, nrhs, &rhs_B, grid, A, m_A, alpha+p[k], combined_localY);

        // printf("First RHS in adi_ls is\n");
        // for (i = 0; i < local_b*nrhs; ++i) {
        //     for (j = 0; j < m_A; ++j) {
        //         printf("%f ", combined_localY[j*local_b*nrhs+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        dScalePermstructInit(m_B, m_B, &ScalePermstruct_B);
        dLUstructInit(m_B, &LUstruct_B);
        PStatInit(&stat_B);
        pdgssvx(&options, &B, &ScalePermstruct_B, rhs_B, local_b, m_A*nrhs, grid,
            &LUstruct_B, &SOLVEstruct_B, berr_B, &stat_B, &info);
        dgather_X(rhs_B, local_b, Y, m_B, m_A*nrhs, grid);

        Destroy_CompRowLoc_Matrix_dist(&B);
        dScalePermstructFree(&ScalePermstruct_B);
        dDestroy_LU(m_B, grid, &LUstruct_B);
        dLUstructFree(&LUstruct_B);
        dSolveFinalize(&options, &SOLVEstruct_B);

        // printf("Global rank %d reaches here b %d.\n", global_rank, k);
        // fflush(stdout);
        MPI_Barrier(grid->comm);

        for (j = 0; j < m_A; ++j) {
            for (i = 0; i < m_A; ++i) {
                newA[j*m_A+i] = A[j*m_A+i] - (i == j ? q[k]+alpha : 0.0);
            }
        }
        dcreate_RHS_ls_sp(F, local_b, nrhs, &rhs_A, grid, m_A, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, -beta+q[k], Y);

        dgetrf_(&m_A, &m_A, newA, &m_A, ipiv, &info);
        dgetrs_("N", &m_A, &n_solveA, newA, &m_A, ipiv, rhs_A, &m_A, &info);
        for (w = 0; w < nrhs; ++w) {
            for (j = 0; j < m_A; ++j) {
                for (i = 0; i < local_b; ++i) {
                    combined_localY[j*nrhs*local_b+w*local_b+i] = rhs_A[w*m_A*local_b+i*m_A+j];
                }
            }
        }
        
        MPI_Barrier(grid->comm);
    }

    for (j = 0; j < m_A*local_b*nrhs; ++j) {
        X[j] = rhs_A[j];
    }

    PStatFree(&stat_B);
    SUPERLU_FREE(berr_B);
    SUPERLU_FREE(Y);
    SUPERLU_FREE(rhs_B);
    SUPERLU_FREE(combined_localY);
    SUPERLU_FREE(newA);
    SUPERLU_FREE(ipiv);
}

void adi_ls2(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *F, int ldf, double *F_transpose, int ldft, 
    int nrhs, double s, double la, double ua, double lb, double ub, double *X, int *grid_proc, int grA, int grB)
{
    SuperMatrix A, B;
    double *rhs_A, *rhs_B;
    double *Y, *combined_localY;
    int ldx, info;
    double *p, *q;
    double alpha, beta;
    int_t l;
    int_t i, j, k, w;
    double *berr_A, *berr_B;
    SuperLUStat_t stat_A, stat_B;
    dScalePermstruct_t ScalePermstruct_A, ScalePermstruct_B;
    dLUstruct_t LUstruct_A, LUstruct_B;
    dSOLVEstruct_t SOLVEstruct_A, SOLVEstruct_B;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    ldx = m_A;

    if (grid_A->iam != -1) {
        if ( !(berr_A = doubleMalloc_dist(m_B*nrhs)) )
            ABORT("Malloc fails for berr_A[].");

    }
    if (grid_B->iam != -1) {
        if ( !(berr_B = doubleMalloc_dist(m_A*nrhs)) )
            ABORT("Malloc fails for berr_B[].");
        if ( !(Y = doubleMalloc_dist(m_B*m_A*nrhs)) )
            ABORT("Malloc fails for Y.");
    }

    alpha = s/2.0;
    beta = s/2.0;
    if (grid_A->iam == 0) {
        if (ua < lb)  {
            if (ua >= lb + s) {
                printf("ua is %f, lb is %f, s is %f.\n", ua, lb, s);
                fflush(stdout);
                ABORT("Inner ADI not ideal for solving.");
            }
        }
        else if (ub < la) {
            if (ub >= la - s) {
                printf("ub is %f, la is %f, s is %f.\n", ub, la, s);
                fflush(stdout);
                ABORT("Inner ADI not ideal for solving.");
            }
        }
    }

    dgenerate_shifts_twogrids(la-alpha, ua-alpha, lb+beta, ub+beta, &p, &q, &l, grid_A, grid_B, grid_proc, grA, grB);

    for (k = 0; k < l; ++k) {
        if (grid_B->iam != -1) {
            dcreate_SuperMatrix(&B, grid_B, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, -beta+p[k]);
        }
        // printf("Global rank %d reaches here aa %d.\n", global_rank, k);
        // fflush(stdout);
        // MPI_Barrier(MPI_COMM_WORLD);

        dcreate_RHS_multiple(F_transpose, ldft, nrhs, &rhs_B, 1, grid_A, grid_B, 
            m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, alpha+p[k], X, m_A, m_B, grid_proc, grA, grB);

        // printf("Global rank %d reaches here a %d.\n", global_rank, k);
        // fflush(stdout);
        // MPI_Barrier(MPI_COMM_WORLD);

        if (grid_B->iam != -1) {
            dScalePermstructInit(m_B, m_B, &ScalePermstruct_B);
            dLUstructInit(m_B, &LUstruct_B);
            PStatInit(&stat_B);
            pdgssvx(&options, &B, &ScalePermstruct_B, rhs_B, ldft, m_A*nrhs, grid_B,
                &LUstruct_B, &SOLVEstruct_B, berr_B, &stat_B, &info);
            dgather_X(rhs_B, ldft, Y, m_B, m_A*nrhs, grid_B);

            Destroy_CompRowLoc_Matrix_dist(&B);
            dScalePermstructFree(&ScalePermstruct_B);
            dDestroy_LU(m_B, grid_B, &LUstruct_B);
            dLUstructFree(&LUstruct_B);
            dSolveFinalize(&options, &SOLVEstruct_B);

            MPI_Barrier(grid_B->comm);
        }
        if (grid_A->iam != -1) {
            MPI_Barrier(grid_A->comm);
        }

        // printf("Global rank %d reaches here b %d.\n", global_rank, k);
        // fflush(stdout);

        if (grid_A->iam != -1) {
            dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, alpha+q[k]);
        }  
        dcreate_RHS_multiple(F, ldf, nrhs, &rhs_A, 0, grid_A, grid_B, 
            m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, -beta+q[k], Y, m_B, m_A, grid_proc, grA, grB);

        // MPI_Barrier(MPI_COMM_WORLD);

        if (grid_A->iam != -1) {
            dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
            dLUstructInit(m_A, &LUstruct_A);
            PStatInit(&stat_A);
            pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldf, m_B*nrhs, grid_A,
                &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);
            dgather_X(rhs_A, ldf, X, m_A, m_B, grid_A);

            Destroy_CompRowLoc_Matrix_dist(&A);
            dScalePermstructFree(&ScalePermstruct_A);
            dDestroy_LU(m_A, grid_A, &LUstruct_A);
            dLUstructFree(&LUstruct_A);
            dSolveFinalize(&options, &SOLVEstruct_A);

            MPI_Barrier(grid_A->comm);
        }
        if (grid_B->iam != -1) {
            MPI_Barrier(grid_B->comm);
        }

        // printf("Global rank %d reaches here d %d.\n", global_rank, k);
        // fflush(stdout);
    }

    if ( grid_A->iam != -1 ) {
        PStatFree(&stat_A);
        SUPERLU_FREE(berr_A);
        SUPERLU_FREE(rhs_A);
    }
    if ( grid_B->iam != -1 ) {
        PStatFree(&stat_B);
        SUPERLU_FREE(berr_B);
        SUPERLU_FREE(Y);
        SUPERLU_FREE(rhs_B);
    }
}

void fadi(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    gridinfo_t *grid_A, gridinfo_t *grid_B, double *U, int ldu, double *V, int ldv,
    double *p, double *q, int_t l, double tol, double **Z, double **D, double **Y, int r, int *rank)
{
    SuperMatrix A, B;
    double *rhs_A, *rhs_B;
    double *localZ, *localY, *localD;
    double *iterZ, *iterY;
    int ldz, ldy, info;
    int ldlz, ldly;
    int rr;
    int_t i, j, k;
    double *berr_A, *berr_B;
    SuperLUStat_t stat_A, stat_B;
    dScalePermstruct_t ScalePermstruct_A, ScalePermstruct_B;
    dLUstruct_t LUstruct_A, LUstruct_B;
    dSOLVEstruct_t SOLVEstruct_A, SOLVEstruct_B;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    ldz = m_A;
    ldy = m_B;
    ldlz = ldu;
    ldly = ldv;

    if (grid_A->iam == 0) {
        if ( !(localD = doubleMalloc_dist(r*l)) )
            ABORT("Malloc fails for localD[].");
    }
    if (grid_A->iam != -1) {
        if ( !(berr_A = doubleMalloc_dist(m_B)) )
            ABORT("Malloc fails for berr_A[].");
        if ( !(rhs_A = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for rhs_A[].");
        if ( !(localZ = doubleMalloc_dist(ldu*r*l)) )
            ABORT("Malloc fails for localZ[].");
        if ( !(iterZ = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for iterZ[].");
    }
    if (grid_B->iam != -1) {
        if ( !(berr_B = doubleMalloc_dist(m_A)) )
            ABORT("Malloc fails for berr_B[].");
        if ( !(rhs_B = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for rhs_B[].");
        if ( !(localY = doubleMalloc_dist(ldv*r*l)) )
            ABORT("Malloc fails for localY[].");
        if ( !(iterY = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for iterY[].");
    }

    if (grid_A->iam != -1) {
        dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[0]);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                rhs_A[j*ldu+i] = U[j*ldu+i];
            }
        }

        // if (grid_A->iam == 0) {
        //     printf("rhs_A before solving is\n");
        //     for (i = 0; i < ldu; ++i) {
        //         for (j = 0; j < r; ++j) {
        //             printf("%f ", rhs_A[j*ldu+i]);
        //         }
        //         printf("\n");
        //     }
        // }

        dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
        dLUstructInit(m_A, &LUstruct_A);
        PStatInit(&stat_A);
        pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldlz, r, grid_A,
            &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

        // if (grid_A->iam == 0) {
        //     printf("rhs_A after solving is\n");
        //     for (i = 0; i < ldu; ++i) {
        //         for (j = 0; j < r; ++j) {
        //             printf("%f ", rhs_A[j*ldu+i]);
        //         }
        //         printf("\n");
        //     }
        // }

        Destroy_CompRowLoc_Matrix_dist(&A);
        dScalePermstructFree(&ScalePermstruct_A);
        dDestroy_LU(m_A, grid_A, &LUstruct_A);
        dLUstructFree(&LUstruct_A);
        dSolveFinalize(&options, &SOLVEstruct_A);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                localZ[j*ldu+i] = rhs_A[j*ldu+i];
                iterZ[j*ldu+i] = rhs_A[j*ldu+i];
            }
        }
    }
    if (grid_B->iam != -1) {
        dcreate_SuperMatrix(&B, grid_B, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, p[0]);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldv; ++i) {
                rhs_B[j*ldv+i] = V[j*ldv+i];
            }
        }

        dScalePermstructInit(m_B, m_B, &ScalePermstruct_B);
        dLUstructInit(m_B, &LUstruct_B);
        PStatInit(&stat_B);
        pdgssvx(&options, &B, &ScalePermstruct_B, rhs_B, ldly, r, grid_B,
            &LUstruct_B, &SOLVEstruct_B, berr_B, &stat_B, &info);

        Destroy_CompRowLoc_Matrix_dist(&B);
        dScalePermstructFree(&ScalePermstruct_B);
        dDestroy_LU(m_B, grid_B, &LUstruct_B);
        dLUstructFree(&LUstruct_B);
        dSolveFinalize(&options, &SOLVEstruct_B);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldv; ++i) {
                localY[j*ldv+i] = rhs_B[j*ldv+i];
                iterY[j*ldv+i] = rhs_B[j*ldv+i];
            }
        }
    }
    if (grid_A->iam == 0) {
        for (j = 0; j < r; ++j) {
            localD[j] = q[0] - p[0];
        }
    }
    // if (grid_A->iam == 0) {
    //     printf("IterZ at the current step is\n");
    //     for (i = 0; i < ldu; ++i) {
    //         for (j = 0; j < r; ++j) {
    //             printf("%f ", iterZ[j*ldu+i]);
    //         }
    //         printf("\n");
    //     }
    //     printf("SolutionZ at the current step before recompression is\n");
    //     for (i = 0; i < ldu; ++i) {
    //         for (j = 0; j < r+rr; ++j) {
    //             printf("%f ", localZ[j*ldu+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // printf("Global rank %d gets pre iteration solve done.\n", global_rank);
    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    rr = r;
    for (k = 1; k < l; ++k) {
        if (grid_A->iam != -1) {
            dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[k]);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    rhs_A[j*ldu+i] = (q[k]-p[k-1])*iterZ[j*ldu+i];
                }
            }

            dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
            dLUstructInit(m_A, &LUstruct_A);
            PStatInit(&stat_A);
            pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldlz, r, grid_A,
                &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

            Destroy_CompRowLoc_Matrix_dist(&A);
            dScalePermstructFree(&ScalePermstruct_A);
            dDestroy_LU(m_A, grid_A, &LUstruct_A);
            dLUstructFree(&LUstruct_A);
            dSolveFinalize(&options, &SOLVEstruct_A);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    iterZ[j*ldu+i] += rhs_A[j*ldu+i];
                    localZ[rr*ldu+j*ldu+i] = iterZ[j*ldu+i];
                }
            }
        }
        if (grid_B->iam != -1) {
            dcreate_SuperMatrix(&B, grid_B, m_B, m_B, nnz_B, nzval_B, rowind_B, colptr_B, p[k]);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldv; ++i) {
                    rhs_B[j*ldv+i] = (p[k]-q[k-1])*iterY[j*ldv+i];
                }
            }

            dScalePermstructInit(m_B, m_B, &ScalePermstruct_B);
            dLUstructInit(m_B, &LUstruct_B);
            PStatInit(&stat_B);
            pdgssvx(&options, &B, &ScalePermstruct_B, rhs_B, ldly, r, grid_B,
                &LUstruct_B, &SOLVEstruct_B, berr_B, &stat_B, &info);

            Destroy_CompRowLoc_Matrix_dist(&B);
            dScalePermstructFree(&ScalePermstruct_B);
            dDestroy_LU(m_B, grid_B, &LUstruct_B);
            dLUstructFree(&LUstruct_B);
            dSolveFinalize(&options, &SOLVEstruct_B);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldv; ++i) {
                    iterY[j*ldv+i] += rhs_B[j*ldv+i];
                    localY[rr*ldv+j*ldv+i] = iterY[j*ldv+i];
                }
            }
        }
        if (grid_A->iam == 0) {
            for (j = 0; j < r; ++j) {
                localD[rr+j] = q[k] - p[k];
            }
        }

        // printf("Global rank %d gets solve for iteration %d done.\n", global_rank, k);
        // printf("Global rank %d ready to recompress for iteration %d, with precompressed rank %d.\n", global_rank, k, rr+r);
        // if (grid_A->iam == 0) {
        //     printf("IterZ at the current step is\n");
        //     for (i = 0; i < ldu; ++i) {
        //         for (j = 0; j < r; ++j) {
        //             printf("%f ", iterZ[j*ldu+i]);
        //         }
        //         printf("\n");
        //     }
        //     printf("SolutionZ at the current step before recompression is\n");
        //     for (i = 0; i < ldu; ++i) {
        //         for (j = 0; j < r+rr; ++j) {
        //             printf("%f ", localZ[j*ldu+i]);
        //         }
        //         printf("\n");
        //     }
        // }
        // if (grid_B->iam == 0) {
        //     printf("IterY at the current step is\n");
        //     for (i = 0; i < ldv; ++i) {
        //         for (j = 0; j < r; ++j) {
        //             printf("%f ", iterY[j*ldv+i]);
        //         }
        //         printf("\n");
        //     }
        //     printf("SolutionY at the current step before recompression is\n");
        //     for (i = 0; i < ldv; ++i) {
        //         for (j = 0; j < r+rr; ++j) {
        //             printf("%f ", localY[j*ldv+i]);
        //         }
        //         printf("\n");
        //     }
        // }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        rr += r;
        double *compressU, *compressS, *compressV;
        drecompression_dist(localZ, ldu, m_A, &compressU, localD, &compressS, grid_A,
            localY, ldv, m_B, &compressV, grid_B, &rr, tol);
        if (grid_A->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldu; ++i) {
                    localZ[j*ldu+i] = compressU[j*ldu+i];
                }
            }
            SUPERLU_FREE(compressU);
        }
        if (grid_B->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldv; ++i) {
                    localY[j*ldv+i] = compressV[j*ldv+i];
                }
            }
            SUPERLU_FREE(compressV);
        }
        if (grid_A->iam == 0) {
            for (j = 0; j < rr; ++j) {
                localD[j] = compressS[j];
            }
            SUPERLU_FREE(compressS);
        }

        // printf("Global rank %d gets recompression for iteration %d done.\n", global_rank, k);
        // fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    *rank = rr;
    if (grid_A->iam == 0) {
        if ( !(*Z = doubleMalloc_dist(m_A*rr)) )
            ABORT("Malloc fails for *Z[].");
        if ( !(*D = doubleMalloc_dist(rr)) )
            ABORT("Malloc fails for *D[].");
        for (j = 0; j < rr; ++j) {
            (*D)[j] = localD[j];
        }
    }
    if (grid_A->iam != -1) {
        dgather_X(localZ, ldu, *Z, m_A, rr, grid_A);
    }
    if (grid_B->iam == 0) {
        if ( !(*Y = doubleMalloc_dist(m_B*rr)) )
            ABORT("Malloc fails for *Y[].");
    }
    if (grid_B->iam != -1) {
        dgather_X(localY, ldv, *Y, m_B, rr, grid_B);
    }

    if ( grid_A->iam != -1 ) {
        PStatFree(&stat_A);
        SUPERLU_FREE(berr_A);
        SUPERLU_FREE(rhs_A);
        SUPERLU_FREE(localZ);
        SUPERLU_FREE(iterZ);
    }
    if ( grid_B->iam != -1 ) {
        PStatFree(&stat_B);
        SUPERLU_FREE(berr_B);
        SUPERLU_FREE(rhs_B);
        SUPERLU_FREE(localY);
        SUPERLU_FREE(iterY);
    }
    if (grid_A->iam == 0) {
        SUPERLU_FREE(localD);
    }

    // if (grid_A->iam == 0) {
    //     printf("Solution Z is\n");
    //     for (i = 0; i < m_A; ++i) {
    //         for (j = 0; j < rr; ++j) {
    //             printf("%f ", (*Z)[j*m_A+i]);
    //         }
    //         printf("\n");
    //     }
    //     printf("Solution D is\n");
    //     for (j = 0; j < rr; ++j) {
    //         printf("%f ", (*D)[j]);
    //     }
    //     printf("\n");
    // }
    // if (grid_B->iam == 0) {
    //     printf("Solution Y is\n");
    //     for (i = 0; i < m_B; ++i) {
    //         for (j = 0; j < rr; ++j) {
    //             printf("%f ", (*Y)[j*m_B+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // fflush(stdout);
}

void fadi_col(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    gridinfo_t *grid_A, double *U, int ldu, double *p, double *q, int_t l, double tol, double **Z, int r, int *rank)
{
    SuperMatrix A;
    double *rhs_A;
    double *localZ;
    double *iterZ;
    int info;
    int ldlz;
    int rr, ovsamp;
    int_t i, j, k;
    double *berr_A;
    SuperLUStat_t stat_A;
    dScalePermstruct_t ScalePermstruct_A;
    dLUstruct_t LUstruct_A;
    dSOLVEstruct_t SOLVEstruct_A;

    ldlz = ldu;

    if ( !(berr_A = doubleMalloc_dist(r)) )
        ABORT("Malloc fails for berr_A[].");
    if ( !(rhs_A = doubleMalloc_dist(ldu*r)) )
        ABORT("Malloc fails for rhs_A[].");
    if ( !(localZ = doubleMalloc_dist(ldu*r*l)) )
        ABORT("Malloc fails for localZ[].");
    if ( !(iterZ = doubleMalloc_dist(ldu*r)) )
        ABORT("Malloc fails for iterZ[].");

    dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[0]);

    for (j = 0; j < r; ++j) {
        for (i = 0; i < ldu; ++i) {
            rhs_A[j*ldu+i] = U[j*ldu+i];
        }
    }

    dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
    dLUstructInit(m_A, &LUstruct_A);
    PStatInit(&stat_A);
    pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldlz, r, grid_A,
        &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&ScalePermstruct_A);
    dDestroy_LU(m_A, grid_A, &LUstruct_A);
    dLUstructFree(&LUstruct_A);
    dSolveFinalize(&options, &SOLVEstruct_A);

    for (j = 0; j < r; ++j) {
        for (i = 0; i < ldu; ++i) {
            localZ[j*ldu+i] = rhs_A[j*ldu+i]*(q[0]-p[0]);
            iterZ[j*ldu+i] = rhs_A[j*ldu+i];
        }
    }
    MPI_Barrier(grid_A->comm);

    for (k = 1; k < l; ++k) {
        dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[k]);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                rhs_A[j*ldu+i] = (q[k]-p[k-1])*iterZ[j*ldu+i];
            }
        }

        dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
        dLUstructInit(m_A, &LUstruct_A);
        PStatInit(&stat_A);
        pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldlz, r, grid_A,
            &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

        Destroy_CompRowLoc_Matrix_dist(&A);
        dScalePermstructFree(&ScalePermstruct_A);
        dDestroy_LU(m_A, grid_A, &LUstruct_A);
        dLUstructFree(&LUstruct_A);
        dSolveFinalize(&options, &SOLVEstruct_A);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                iterZ[j*ldu+i] += rhs_A[j*ldu+i];
                localZ[k*r*ldu+j*ldu+i] = iterZ[j*ldu+i]*(q[k]-p[k]);
            }
        }
              
        MPI_Barrier(grid_A->comm);
    }

    // printf("Proc %d in grid_A gets T1 before truncation.\n", grid_A->iam);
    // for (i = 0; i < ldu; ++i) {
    //     for (j = 0; j < r*l; ++j) {
    //         printf("%f ", localZ[j*ldu+i]);
    //     }
    //     printf("\n");
    // }
    // fflush(stdout);
    // MPI_Barrier(grid_A->comm);

    dCPQR_dist_getQ(localZ, ldu, Z, r*l, rank, grid_A, tol);
    // ovsamp = r*l >= 20 ? 5 : 2;
    // dCPQR_dist_rand_getQ(localZ, ldu, m_A, Z, r*l, rank, grid_A, tol, ovsamp);

    // printf("Proc %d in grid_A gets T1 after truncation.\n", grid_A->iam);
    // for (i = 0; i < ldu; ++i) {
    //     for (j = 0; j < *rank; ++j) {
    //         printf("%f ", (*Z)[j*ldu+i]);
    //     }
    //     printf("\n");
    // }
    // fflush(stdout);
    MPI_Barrier(grid_A->comm);

    PStatFree(&stat_A);
    SUPERLU_FREE(berr_A);
    SUPERLU_FREE(rhs_A);
    SUPERLU_FREE(localZ);
    SUPERLU_FREE(iterZ);
}

void fadi_col_adils(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B, gridinfo_t *grid, double *F, int local_b,
    double *p, double *q, int_t l, double la, double ua, double lb, double ub, double tol, double **Z, int r, int *rank)
{
    double *rhs_B;
    double *localZ, *iterZ, *tmp_iterZ;
    int rr, ldu, ovsamp;
    int_t i, j, k;
    double *nzval_B_neg;
    int_t  *rowind_B_neg, *colptr_B_neg;

    ldu = m_A*local_b;

    dallocateA_dist(m_B, nnz_B, &nzval_B_neg, &rowind_B_neg, &colptr_B_neg);
    for (i = 0; i < m_B; ++i) {
        for (j = colptr_B[i]; j < colptr_B[i+1]; ++j) {
            nzval_B_neg[j] = -nzval_B[j];
            rowind_B_neg[j] = rowind_B[j];
        }
        colptr_B_neg[i] = colptr_B[i];
    }
    colptr_B_neg[m_B] = colptr_B[m_B];

    if ( !(rhs_B = doubleMalloc_dist(ldu*r)) )
        ABORT("Malloc fails for rhs_A[].");
    if ( !(localZ = doubleMalloc_dist(ldu*r*l)) )
        ABORT("Malloc fails for localZ[].");
    if ( !(iterZ = doubleMalloc_dist(ldu*r)) )
        ABORT("Malloc fails for iterZ[].");
    if ( !(tmp_iterZ = doubleMalloc_dist(ldu*r)) )
        ABORT("Malloc fails for tmp_iterZ[].");

    for (j = 0; j < r; ++j) {
        for (i = 0; i < ldu; ++i) {
            rhs_B[j*ldu+i] = F[j*ldu+i];
        }
    }
    adi_ls(options, m_A, A, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg, grid, rhs_B, local_b, r, q[0], la, ua, lb, ub, iterZ);
    for (j = 0; j < r; ++j) {
        for (i = 0; i < ldu; ++i) {
            localZ[j*ldu+i] = iterZ[j*ldu+i]*(q[0]-p[0]);
        }
    }
    MPI_Barrier(grid->comm);

    for (k = 1; k < l; ++k) {
        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                rhs_B[j*ldu+i] = (q[k]-p[k-1])*iterZ[j*ldu+i];
            }
        }
        adi_ls(options, m_A, A, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg, grid, 
            rhs_B, local_b, r, q[k], la, ua, lb, ub, tmp_iterZ);
        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                iterZ[j*ldu+i] += tmp_iterZ[j*ldu+i];
                localZ[k*r*ldu+j*ldu+i] = iterZ[j*ldu+i]*(q[k]-p[k]);
            }
        }
              
        MPI_Barrier(grid->comm);
    }

    dCPQR_dist_getQ(localZ, ldu, Z, r*l, rank, grid, tol);
    // ovsamp = r*l >= 20 ? 5 : 2;
    // dCPQR_dist_rand_getQ(localZ, ldu, m_A*m_B, Z, r*l, rank, grid, tol, ovsamp);

    MPI_Barrier(grid->comm);

    SUPERLU_FREE(rhs_B);
    SUPERLU_FREE(localZ);
    SUPERLU_FREE(iterZ);
    SUPERLU_FREE(tmp_iterZ);
}

void fadi_sp(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_B, gridinfo_t *grid_C, double *U, int ldu, double *V, int ldv,
    double *p, double *q, int_t l, double tol, double **Z, double **Y, int r, int *rank,
    double la, double ua, double lb, double ub, int *grid_proc, int grB, int grC)
{
    SuperMatrix B, C;
    double *rhs_B, *rhs_C;
    double *localZ, *localY, *localD, *finalD;
    double *iterZ, *tmp_iterZ, *iterY;
    int ldz, ldy, info;
    int ldlz, ldly;
    int rr;
    int_t i, j, k;
    double *berr_C;
    SuperLUStat_t stat_C;
    dScalePermstruct_t ScalePermstruct_C;
    dLUstruct_t LUstruct_C;
    dSOLVEstruct_t SOLVEstruct_C;

    ldz = ldu;
    ldy = ldv;
    ldlz = ldu / m_A;
    ldly = ldv;

    // printf("Process with id %d in B, and %d in C starts fadi_sp.\n", grid_B->iam, grid_C->iam);
    // fflush(stdout);
    // if (grid_B->iam != -1) {
    //     MPI_Barrier(grid_B->comm);
    // }
    // if (grid_C->iam != -1) {
    //     printf("Process with id %d in C has ldv %d and r %d.\n", grid_C->iam, ldv, r);
    //     MPI_Barrier(grid_C->comm);
    // }

    if (grid_B->iam == 0) {
        if ( !(localD = doubleMalloc_dist(r*l)) )
            ABORT("Malloc fails for localD[].");
    }
    if (grid_B->iam != -1) {
        if ( !(rhs_B = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for rhs_B[].");
        if ( !(localZ = doubleMalloc_dist(ldu*r*l)) )
            ABORT("Malloc fails for localZ[].");
        if ( !(iterZ = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for iterZ[].");
        if ( !(tmp_iterZ = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for tmp_iterZ[].");
    }
    if (grid_C->iam != -1) {
        if ( !(berr_C = doubleMalloc_dist(r)) )
            ABORT("Malloc fails for berr_C[].");
        if ( !(rhs_C = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for rhs_C[].");
        if ( !(localY = doubleMalloc_dist(ldv*r*l)) )
            ABORT("Malloc fails for localY[].");
        if ( !(iterY = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for iterY[].");
    }

    if (grid_B->iam != -1) {
        // printf("U[0] is %f\n", U[0]);
        // for (i = 0; i < ldu; ++i) {
        //     for (j = 0; j < r; ++j) {
        //         printf("%f ", U[j*ldu+i]);
        //     }
        //     printf("\n");
        // }
        // printf("U has size %d and %d on proc %d.\n", ldu, r, grid_B->iam);
        // fflush(stdout);
        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                rhs_B[j*ldu+i] = U[j*ldu+i];
            }
        }

        // printf("First rhs for adi_ls is\n");
        // for (i = 0; i < ldu; ++i) {
        //     for (j = 0; j < r; ++j) {
        //         printf("%f ", rhs_B[j*ldu+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        adi_ls(options, m_A, A, m_B, nnz_B, nzval_B, rowind_B, colptr_B, grid_B, rhs_B, ldlz, r, q[0], la, ua, lb, ub, iterZ);

        // if (grid_B->iam == 0) {
        //     printf("Finish first adi_ls.\n");
        //     fflush(stdout);
        // }
        // printf("proc %d computes first Z.\n", grid_B->iam);
        // for (i = 0; i < ldu; ++i) {
        //     for (j = 0; j < r; ++j) {
        //         printf("%f ", iterZ[j*ldu+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                localZ[j*ldu+i] = iterZ[j*ldu+i];
            }
        }
    }
    if (grid_C->iam != -1) {
        // printf("proc %d on C starts first step.\n", grid_C->iam);
        // for (i = 0; i < ldv; ++i) {
        //     for (j = 0; j < r; ++j) {
        //         printf("%f ", V[j*ldv+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldv; ++i) {
                rhs_C[j*ldv+i] = V[j*ldv+i];
            }
        }

        printf("First RHS on proc %d in C is\n", grid_C->iam);
        for (i = 0; i < ldv; ++i) {
            for (j = 0; j < r; ++j) {
                printf("%f ", rhs_C[j*ldv+i]);
            }
            printf("\n");
        }
        fflush(stdout);

        // if (grid_C->iam == 0) {
        //     printf("nnz is %d.\n", nnz_C);
        //     printf("colptr is ");
        //     for (i = 0; i < m_C+1; ++i) {
        //         printf("%d ", colptr_C[i]);
        //     }
        //     printf("nzval is ");
        //     for (i = 0; i < m_C; ++i) {
        //         for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
        //             printf("%f ", nzval_C[j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("rowind is ");
        //     for (i = 0; i < m_C; ++i) {
        //         for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
        //             printf("%d ", rowind_C[j]);
        //         }
        //         printf("\n");
        //     }
        //     fflush(stdout);
        // }

        dcreate_SuperMatrix(&C, grid_C, m_C, m_C, nnz_C, nzval_C, rowind_C, colptr_C, p[0]);

        // printf("proc %d on C is here.\n", grid_C->iam);
        // fflush(stdout);

        dScalePermstructInit(m_C, m_C, &ScalePermstruct_C);
        dLUstructInit(m_C, &LUstruct_C);
        PStatInit(&stat_C);
        pdgssvx(&options, &C, &ScalePermstruct_C, rhs_C, ldly, r, grid_C,
            &LUstruct_C, &SOLVEstruct_C, berr_C, &stat_C, &info);

        printf("proc %d on C is here.\n", grid_C->iam);
        fflush(stdout);

        Destroy_CompRowLoc_Matrix_dist(&C);
        dScalePermstructFree(&ScalePermstruct_C);
        dDestroy_LU(m_C, grid_C, &LUstruct_C);
        dLUstructFree(&LUstruct_C);
        dSolveFinalize(&options, &SOLVEstruct_C);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldv; ++i) {
                localY[j*ldv+i] = rhs_C[j*ldv+i];
                iterY[j*ldv+i] = rhs_C[j*ldv+i];
            }
        }
    }
    if (grid_B->iam == 0) {
        for (j = 0; j < r; ++j) {
            localD[j] = q[0] - p[0];
        }
    }
    
    if (grid_B->iam != -1) {
        MPI_Barrier(grid_B->comm);
    }
    if (grid_C->iam != -1) {
        MPI_Barrier(grid_C->comm);
    }

    rr = r;
    for (k = 1; k < l; ++k) {
        if (grid_B->iam != -1) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    rhs_B[j*ldu+i] = (q[k]-p[k-1])*iterZ[j*ldu+i];
                }
            }

            adi_ls(options, m_A, A, m_B, nnz_B, nzval_B, rowind_B, colptr_B, grid_B, rhs_B, ldlz, r, q[k], la, ua, lb, ub, tmp_iterZ);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    iterZ[j*ldu+i] += tmp_iterZ[j*ldu+i];
                    localZ[rr*ldu+j*ldu+i] = iterZ[j*ldu+i];
                }
            }
        }
        if (grid_C->iam != -1) {
            dcreate_SuperMatrix(&C, grid_C, m_C, m_C, nnz_C, nzval_C, rowind_C, colptr_C, p[k]);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldv; ++i) {
                    rhs_C[j*ldv+i] = (p[k]-q[k-1])*iterY[j*ldv+i];
                }
            }

            dScalePermstructInit(m_C, m_C, &ScalePermstruct_C);
            dLUstructInit(m_C, &LUstruct_C);
            PStatInit(&stat_C);
            pdgssvx(&options, &C, &ScalePermstruct_C, rhs_C, ldly, r, grid_C,
                &LUstruct_C, &SOLVEstruct_C, berr_C, &stat_C, &info);

            Destroy_CompRowLoc_Matrix_dist(&C);
            dScalePermstructFree(&ScalePermstruct_C);
            dDestroy_LU(m_C, grid_C, &LUstruct_C);
            dLUstructFree(&LUstruct_C);
            dSolveFinalize(&options, &SOLVEstruct_C);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldv; ++i) {
                    iterY[j*ldv+i] += rhs_C[j*ldv+i];
                    localY[rr*ldv+j*ldv+i] = iterY[j*ldv+i];
                }
            }
        }
        if (grid_B->iam == 0) {
            for (j = 0; j < r; ++j) {
                localD[rr+j] = q[k] - p[k];
            }
        }

        // if (grid_B->iam == 0) {
        //     printf("Start recompression for iteration %d.\n", k);
        //     fflush(stdout);
        // }

        rr += r;

        // if (grid_B->iam != -1) {
        //     printf("proc %d in B goes into recompression for iteration %d with \n", grid_B->iam, k);
        //     for (i = 0; i < ldu; ++i) {
        //         for (j = 0; j < rr; ++j) {
        //             printf("%f ", localZ[j*ldu+i]);
        //         }
        //         printf("\n");
        //     }
        //     fflush(stdout);
        // }
        // if (grid_C->iam != -1) {
        //     printf("proc %d in C goes into recompression for iteration %d with \n", grid_C->iam, k);
        //     for (i = 0; i < ldv; ++i) {
        //         for (j = 0; j < rr; ++j) {
        //             printf("%f ", localY[j*ldv+i]);
        //         }
        //         printf("\n");
        //     }
        //     fflush(stdout);
        // }

        if (grid_B->iam != -1) {
            MPI_Barrier(grid_B->comm);
        }
        if (grid_C->iam != -1) {
            MPI_Barrier(grid_C->comm);
        }

        double *compressU, *compressS, *compressV;
        drecompression_dist_twogrids(localZ, ldu, m_A*m_B, &compressU, localD, &compressS, grid_B,
            localY, ldv, m_C, &compressV, grid_C, &rr, tol, grid_proc, grB, grC);
        if (grid_B->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldu; ++i) {
                    localZ[j*ldu+i] = compressU[j*ldu+i];
                }
            }
            SUPERLU_FREE(compressU);
        }
        if (grid_C->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldv; ++i) {
                    localY[j*ldv+i] = compressV[j*ldv+i];
                }
            }
            SUPERLU_FREE(compressV);
        }
        if (grid_B->iam == 0) {
            for (j = 0; j < rr; ++j) {
                localD[j] = compressS[j];
            }
            SUPERLU_FREE(compressS);
        }

        // printf("proc %d in B and %d in C gets recompression for iteration %d done.\n", grid_B->iam, grid_C->iam, k);
        fflush(stdout);
        if (grid_B->iam != -1) {
            MPI_Barrier(grid_B->comm);
        }
        if (grid_C->iam != -1) {
            MPI_Barrier(grid_C->comm);
        }
    }

    // printf("proc %d in B and %d in C gets local factorization with rank %d.\n", grid_B->iam, grid_C->iam, rr);
    // if (grid_B->iam != -1) {
    //     printf("proc %d in B gets localZ\n", grid_B->iam);
    //     for (i = 0; i < ldu; ++i) {
    //         for (j = 0; j < rr; ++j) {
    //             printf("%f ", localZ[j*ldu+i]);
    //         }
    //         printf("\n");
    //     }

    //     if (grid_B->iam == 0) {
    //         printf("localD is\n");
    //         for (j = 0; j < rr; ++j) {
    //             printf("%f ", localD[j]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }
    // if (grid_C->iam != -1) {
    //     printf("proc %d in C gets localY\n", grid_C->iam);
    //     for (i = 0; i < ldv; ++i) {
    //         for (j = 0; j < rr; ++j) {
    //             printf("%f ", localY[j*ldv+i]);
    //         }
    //         printf("\n");
    //     }
    //     fflush(stdout);
    // }
    fflush(stdout);

    *rank = rr;
    if (grid_B->iam != -1) {
        if ( !(finalD = doubleMalloc_dist(rr)) )
            ABORT("Malloc fails for finalD[].");
        if (grid_B->iam == 0) {
            for (j = 0; j < rr; ++j) {
                finalD[j] = localD[j];
            }
        }
        MPI_Bcast(finalD, rr, MPI_DOUBLE, 0, grid_B->comm);

        // printf("proc %d in B gets finalD ", grid_B->iam);
        // for (j = 0; j < rr; ++j) {
        //     printf("%f ", finalD[j]);
        // }
        // printf("\n");
        // fflush(stdout);
    
        if ( !(*Z = doubleMalloc_dist(ldu*rr)) )
            ABORT("Malloc fails for *Z[].");

        for (j = 0; j < rr; ++j) {
            for (i = 0; i < ldu; ++i) {
                (*Z)[j*ldu+i] = localZ[j*ldu+i]*finalD[j];
            }
        }

        // printf("proc %d in B gets finalZ ", grid_B->iam);
        // for (i = 0; i < ldu; ++i) {
        //     for (j = 0; j < rr; ++j) {
        //         printf("%f ", (*Z)[j*ldu+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
    }
    
    if (grid_C->iam != -1) {
        if ( !(*Y = doubleMalloc_dist(ldv*rr)) )
            ABORT("Malloc fails for *Y[].");

        for (j = 0; j < ldv; ++j) {
            for (i = 0; i < rr; ++i) {
                (*Y)[j*rr+i] = localY[i*ldv+j];
            }
        }
    }

    if ( grid_B->iam != -1 ) {
        SUPERLU_FREE(rhs_B);
        SUPERLU_FREE(localZ);
        SUPERLU_FREE(iterZ);
        SUPERLU_FREE(tmp_iterZ);
        SUPERLU_FREE(finalD);
    }
    if ( grid_C->iam != -1 ) {
        PStatFree(&stat_C);
        SUPERLU_FREE(berr_C);
        SUPERLU_FREE(rhs_C);
        SUPERLU_FREE(localY);
        SUPERLU_FREE(iterY);
    }
    if (grid_B->iam == 0) {
        SUPERLU_FREE(localD);
    }
}

void fadi_sp_2sided(superlu_dist_options_t options, int_t m_A, double *A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C, int_t m_D, double *D,
    gridinfo_t *grid_B, gridinfo_t *grid_C, double *U, int ldu, double *V, int ldv,
    double *p, double *q, int_t l, double tol, double **Z, double **Y, int r, int *rank,
    double la, double ua, double lb, double ub, double lc, double uc, double ld, double ud, int *grid_proc, int grB, int grC)
{
    SuperMatrix B, C;
    double *rhs_B, *rhs_C;
    double *localZ, *localY, *localS, *finalS;
    double *iterZ, *tmp_iterZ, *iterY, *tmp_iterY;
    int ldz, ldy, info;
    int ldlz, ldly;
    int rr;
    int_t i, j, k;
    
    ldz = ldu;
    ldy = ldv;
    ldlz = ldu / m_A;
    ldly = ldv / m_D;

    if (grid_B->iam == 0) {
        if ( !(localS = doubleMalloc_dist(r*l)) )
            ABORT("Malloc fails for localD[].");
    }
    if (grid_B->iam != -1) {
        if ( !(rhs_B = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for rhs_B[].");
        if ( !(localZ = doubleMalloc_dist(ldu*r*l)) )
            ABORT("Malloc fails for localZ[].");
        if ( !(iterZ = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for iterZ[].");
        if ( !(tmp_iterZ = doubleMalloc_dist(ldu*r)) )
            ABORT("Malloc fails for tmp_iterZ[].");
    }
    if (grid_C->iam != -1) {
        if ( !(rhs_C = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for rhs_C[].");
        if ( !(localY = doubleMalloc_dist(ldv*r*l)) )
            ABORT("Malloc fails for localY[].");
        if ( !(iterY = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for iterY[].");
        if ( !(tmp_iterY = doubleMalloc_dist(ldv*r)) )
            ABORT("Malloc fails for tmp_iterY[].");
    }

    if (grid_B->iam != -1) {
        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                rhs_B[j*ldu+i] = U[j*ldu+i];
            }
        }

        adi_ls(options, m_A, A, m_B, nnz_B, nzval_B, rowind_B, colptr_B, grid_B, rhs_B, ldlz, r, q[0], la, ua, lb, ub, iterZ);

        for (j = 0; j < r; ++j) {
            for (i = 0; i < ldu; ++i) {
                localZ[j*ldu+i] = iterZ[j*ldu+i];
            }
        }
    }
    if (grid_C->iam != -1) {
        for (k = 0; k < r; ++k) {
            for (j = 0; j < m_D; ++j) {
                for (i = 0; i < ldly; ++i) {
                    rhs_C[k*ldv+j*ldly+i] = -V[k*ldv+i*m_D+j];
                }
            }
        }

        adi_ls(options, m_D, D, m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid_C, rhs_C, ldly, r, p[0], ld, ud, lc, uc, iterY);

        for (k = 0; k < r; ++k) {
            for (j = 0; j < ldly; ++j) {
                for (i = 0; i < m_D; ++i) {
                    localY[k*ldv+j*m_D+i] = iterY[k*ldv+i*ldly+j];
                }
            }
        }
    }
    if (grid_B->iam == 0) {
        for (j = 0; j < r; ++j) {
            localS[j] = q[0] - p[0];
        }
    }
    
    if (grid_B->iam != -1) {
        MPI_Barrier(grid_B->comm);
    }
    if (grid_C->iam != -1) {
        MPI_Barrier(grid_C->comm);
    }

    rr = r;
    for (k = 1; k < l; ++k) {
        if (grid_B->iam != -1) {
            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    rhs_B[j*ldu+i] = (q[k]-p[k-1])*iterZ[j*ldu+i];
                }
            }

            adi_ls(options, m_A, A, m_B, nnz_B, nzval_B, rowind_B, colptr_B, grid_B, rhs_B, ldlz, r, q[k], la, ua, lb, ub, tmp_iterZ);

            for (j = 0; j < r; ++j) {
                for (i = 0; i < ldu; ++i) {
                    iterZ[j*ldu+i] += tmp_iterZ[j*ldu+i];
                    localZ[rr*ldu+j*ldu+i] = iterZ[j*ldu+i];
                }
            }
        }
        if (grid_C->iam != -1) {
            for (k = 0; k < r; ++k) {
                for (j = 0; j < m_D; ++j) {
                    for (i = 0; i < ldly; ++i) {
                        rhs_C[k*ldy+j*ldly+i] = -(p[k]-q[k-1])*iterY[k*ldy+i*m_D+j];
                    }
                }
            }

            adi_ls(options, m_D, D, m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid_C, rhs_C, ldly, r, p[k], ld, ud, lc, uc, tmp_iterY);

            for (k = 0; k < r; ++k) {
                for (j = 0; j < ldly; ++j) {
                    for (i = 0; i < m_D; ++i) {
                        iterY[k*ldy+i*ldly+j] += tmp_iterY[k*ldy+i*ldly+j];
                        localY[rr*ldv+k*ldv+j*m_D+i] = iterY[k*ldy+i*ldly+j];
                    }
                }
            }
        }
        if (grid_B->iam == 0) {
            for (j = 0; j < r; ++j) {
                localS[rr+j] = q[k] - p[k];
            }

            printf("Grid 1 finishes solving of iteration %d.\n", k);
            fflush(stdout);
        }

        if (grid_C->iam == 0) {
            printf("Grid 2 finishes solving of iteration %d.\n", k);
            fflush(stdout);
        }

        rr += r;

        if (grid_B->iam != -1) {
            MPI_Barrier(grid_B->comm);
        }
        if (grid_C->iam != -1) {
            MPI_Barrier(grid_C->comm);
        }

        double *compressU, *compressS, *compressV;
        drecompression_dist_twogrids(localZ, ldu, m_A*m_B, &compressU, localS, &compressS, grid_B,
            localY, ldv, m_C*m_D, &compressV, grid_C, &rr, tol, grid_proc, grB, grC);
        if (grid_B->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldu; ++i) {
                    localZ[j*ldu+i] = compressU[j*ldu+i];
                }
            }
            SUPERLU_FREE(compressU);
        }
        if (grid_C->iam != -1) {
            for (j = 0; j < rr; ++j) {
                for (i = 0; i < ldv; ++i) {
                    localY[j*ldv+i] = compressV[j*ldv+i];
                }
            }
            SUPERLU_FREE(compressV);
        }
        if (grid_B->iam == 0) {
            for (j = 0; j < rr; ++j) {
                localS[j] = compressS[j];
            }
            SUPERLU_FREE(compressS);

            printf("Grid 1 finishes recompression of iteration %d.\n", k);
            fflush(stdout);
        }
        if (grid_C->iam == 0) {
            printf("Grid 2 finishes recompression of iteration %d.\n", k);
            fflush(stdout);
        }

        // printf("proc %d in B and %d in C gets recompression for iteration %d done.\n", grid_B->iam, grid_C->iam, k);
        fflush(stdout);
        if (grid_B->iam != -1) {
            MPI_Barrier(grid_B->comm);
        }
        if (grid_C->iam != -1) {
            MPI_Barrier(grid_C->comm);
        }
    }

    *rank = rr;
    if (grid_B->iam != -1) {
        if ( !(finalS = doubleMalloc_dist(rr)) )
            ABORT("Malloc fails for finalD[].");
        if (grid_B->iam == 0) {
            for (j = 0; j < rr; ++j) {
                finalS[j] = localS[j];
            }
        }
        MPI_Bcast(finalS, rr, MPI_DOUBLE, 0, grid_B->comm);

        if ( !(*Z = doubleMalloc_dist(ldu*rr)) )
            ABORT("Malloc fails for *Z[].");

        for (j = 0; j < rr; ++j) {
            for (i = 0; i < ldu; ++i) {
                (*Z)[j*ldu+i] = localZ[j*ldu+i]*finalS[j];
            }
        }
    }
    
    if (grid_C->iam != -1) {
        if ( !(*Y = doubleMalloc_dist(ldv*rr)) )
            ABORT("Malloc fails for *Y[].");

        for (j = 0; j < ldv; ++j) {
            for (i = 0; i < rr; ++i) {
                (*Y)[j*rr+i] = localY[i*ldv+j];
            }
        }
    }

    if ( grid_B->iam != -1 ) {
        SUPERLU_FREE(rhs_B);
        SUPERLU_FREE(localZ);
        SUPERLU_FREE(iterZ);
        SUPERLU_FREE(tmp_iterZ);
        SUPERLU_FREE(finalS);
    }
    if ( grid_C->iam != -1 ) {
        SUPERLU_FREE(rhs_C);
        SUPERLU_FREE(localY);
        SUPERLU_FREE(iterY);
        SUPERLU_FREE(tmp_iterY);
    }
    if (grid_B->iam == 0) {
        SUPERLU_FREE(localS);
    }
}

void fadi_ttsvd_3d(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_A, gridinfo_t *grid_B, gridinfo_t *grid_C, double *U1, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc)
{
    SuperMatrix GA;
    double *newA, *tmpA, *newA_onB, *newU2;
    double *global_T1, *global_T1_onB;
    double *nzval_B_neg, *nzval_C_neg;
    int_t  *rowind_B_neg, *colptr_B_neg, *rowind_C_neg, *colptr_C_neg;
    int rr1, rr2, ldlu2 = ldu2/m_A;
    int ldnewu2 = ldlu2*r2;
    double one = 1.0, zero = 0.0;
    char     transpose[1];
    *transpose = 'N';
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int_t i, j;

    if (grid_A->iam != -1) {
        fadi_col(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, grid_A, U1, ldu1, p1, q1, l1, tol, T1, r1, &rr1);

        if (grid_A->iam == 0) {
            printf("Grid A finishes fadi_col!\n");
            fflush(stdout);

            if ( !(global_T1 = doubleMalloc_dist(m_A*rr1)) )
                ABORT("Malloc fails for global_T1[].");

            if ( !(tmpA = doubleMalloc_dist(m_A*rr1)) )
                ABORT("Malloc fails for tmpA[].");
            if ( !(newA = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newA[].");
        }
        dgather_X(*T1, ldu1, global_T1, m_A, rr1, grid_A);
        if (grid_A->iam == 0) {
            // printf("global_T1 on A is\n");
            // for (i = 0; i < m_A; ++i) {
            //     for (j = 0; j < rr1; ++j) {
            //         printf("%f ", global_T1[j*m_A+i]);
            //     }
            //     printf("\n");
            // }

            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A,
                SLU_NC, SLU_D, SLU_GE);

            sp_dgemm_dist(transpose, rr1, one, &GA, global_T1, m_A, zero, tmpA, m_A);

            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);

            // printf("tmpA on A is\n");
            // for (i = 0; i < m_A; ++i) {
            //     for (j = 0; j < rr1; ++j) {
            //         printf("%f ", tmpA[j*m_A+i]);
            //     }
            //     printf("\n");
            // }

            dgemm_("T", "N", &rr1, &rr1, &m_A, &one, global_T1, &m_A, tmpA, &m_A, &zero, newA, &rr1);

            // printf("newA on A is\n");
            // for (i = 0; i < rr1; ++i) {
            //     for (j = 0; j < rr1; ++j) {
            //         printf("%f ", newA[j*rr1+i]);
            //     }
            //     printf("\n");
            // }
        }

        *rank1 = rr1;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(rank1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rr1 = *rank1;
    
    if (grid_B->iam != -1) {
        if ( !(global_T1_onB = doubleMalloc_dist(m_A*rr1)) )
            ABORT("Malloc fails for global_T1_onB[].");
        if ( !(newA_onB = doubleMalloc_dist(rr1*rr1)) )
            ABORT("Malloc fails for newA_onB[].");
        if ( !(newU2 = doubleMalloc_dist(rr1*ldlu2*r2)) )
            ABORT("Malloc fails for newU2[].");
    }
    transfer_X(global_T1, m_A, rr1, global_T1_onB, grid_A, 1);
    transfer_X(newA, rr1, rr1, newA_onB, grid_A, 1);
    if (grid_B->iam != -1) {
        MPI_Bcast(global_T1_onB, m_A*rr1, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(newA_onB, rr1*rr1, MPI_DOUBLE, 0, grid_B->comm);

        dgemm_("T", "N", &rr1, &ldnewu2, &m_A, &one, global_T1_onB, &m_A, U2, &m_A, &zero, newU2, &rr1);
    }

    // if (grid_B->iam == 0) {
    //     printf("newA on B is\n");
    //     for (i = 0; i < rr1; ++i) {
    //         for (j = 0; j < rr1; ++j) {
    //             printf("%f ", newA_onB[j*rr1+i]);
    //         }
    //         printf("\n");
    //     }
    //     printf("newU2 on B is\n");
    //     for (i = 0; i < rr1*ldlu2; ++i) {
    //         for (j = 0; j < r2; ++j) {
    //             printf("%f ", newU2[j*rr1*ldlu2+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_B->iam != -1) {
        dallocateA_dist(m_B, nnz_B, &nzval_B_neg, &rowind_B_neg, &colptr_B_neg);
        for (i = 0; i < m_B; ++i) {
            for (j = colptr_B[i]; j < colptr_B[i+1]; ++j) {
                nzval_B_neg[j] = -nzval_B[j];
                rowind_B_neg[j] = rowind_B[j];
            }
            colptr_B_neg[i] = colptr_B[i];
        }
        colptr_B_neg[m_B] = colptr_B[m_B];

        // printf("newU2 on proc %d is\n", grid_B->iam);
        // for (i = 0; i < rr1*ldlu2; ++i) {
        //     for (j = 0; j < r2; ++j) {
        //         printf("%f ", newU2[j*rr1*ldlu2+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
    }
    if (grid_C->iam != -1) {
        dallocateA_dist(m_C, nnz_C, &nzval_C_neg, &rowind_C_neg, &colptr_C_neg);
        for (i = 0; i < m_C; ++i) {
            for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
                nzval_C_neg[j] = -nzval_C[j];
                rowind_C_neg[j] = rowind_C[j];
            }
            colptr_C_neg[i] = colptr_C[i];
        }
        colptr_C_neg[m_C] = colptr_C[m_C];

        // if (grid_C->iam == 0) {
        //     printf("nnz is %d.\n", nnz_C);
        //     printf("colptr is ");
        //     for (i = 0; i < m_C+1; ++i) {
        //         printf("%d ", colptr_C[i]);
        //     }
        //     printf("nzval is ");
        //     for (i = 0; i < m_C; ++i) {
        //         for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
        //             printf("%f ", nzval_C[j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("rowind is ");
        //     for (i = 0; i < m_C; ++i) {
        //         for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
        //             printf("%d ", rowind_C[j]);
        //         }
        //         printf("\n");
        //     }
        //     fflush(stdout);
        // }

        // printf("V2 on proc %d is\n", grid_C->iam);
        // for (i = 0; i < ldv2; ++i) {
        //     for (j = 0; j < r2; ++j) {
        //         printf("%f ", V2[j*ldv2+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
    }

    // printf("Process with id %d in A, %d in B, and %d in C gets here.\n", grid_A->iam, grid_B->iam, grid_C->iam);
    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if ((grid_B->iam != -1) || (grid_C->iam != -1)) {
        fadi_sp(options, rr1, newA_onB, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg, 
            m_C, nnz_C, nzval_C_neg, rowind_C_neg, colptr_C_neg, grid_B, grid_C, 
            newU2, rr1*ldlu2, V2, ldv2, p2, q2, l2, tol, T2, T3, r2, rank2, la, ua, lb, ub, grid_proc, 1, 2);

        if (grid_B->iam == 0) {
            printf("Grid B and C finish fadi_sp!\n");
            fflush(stdout);
        }
    }
    MPI_Bcast(rank2, 1, MPI_INT, grid_A->nprow * grid_A->npcol, MPI_COMM_WORLD);

    // if (grid_A->iam != -1) {
    //     printf("Proc %d in grid_A gets T1.\n", grid_A->iam);
    //     for (i = 0; i < ldu1; ++i) {
    //         for (j = 0; j < rr1; ++j) {
    //             printf("%f ", (*T1)[j*ldu1+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // fflush(stdout);

    // if (grid_B->iam != -1) {
    //     printf("Proc %d in grid_B gets T2.\n", grid_B->iam);
    //     for (i = 0; i < rr1*ldlu2; ++i) {
    //         for (j = 0; j < *rank2; ++j) {
    //             printf("%f ", (*T2)[j*rr1*ldlu2+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // fflush(stdout);

    // if (grid_C->iam != -1) {
    //     printf("Proc %d in grid_C gets T3.\n", grid_C->iam);
    //     for (i = 0; i < *rank2; ++i) {
    //         for (j = 0; j < ldv2; ++j) {
    //             printf("%f ", (*T3)[j*(*rank2)+i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (grid_A->iam == 0) {
        SUPERLU_FREE(global_T1);
        SUPERLU_FREE(tmpA);
        SUPERLU_FREE(newA);
    }
    if (grid_B->iam != -1) {
        SUPERLU_FREE(newA_onB);
        SUPERLU_FREE(global_T1_onB);
        SUPERLU_FREE(newU2);
    }
}

void fadi_ttsvd_3d_2grids(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, double *U1, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc, int gr1, int gr2)
{
    SuperMatrix GA;
    double *newA, *tmpA, *newU2;
    double *global_T1;
    double *nzval_A_dup, *nzval_B_neg, *nzval_C_neg;
    int_t  *rowind_A_dup, *colptr_A_dup, *rowind_B_neg, *colptr_B_neg, *rowind_C_neg, *colptr_C_neg;
    int rr1, rr2, ldlu2 = ldu2/m_A;
    int ldnewu2 = ldlu2*r2;
    double one = 1.0, zero = 0.0;
    char     transpose[1];
    *transpose = 'N';
    int root1 = 0, root2 = 0;
    int_t i, j;

    if (grid1->iam != -1) {
        dallocateA_dist(m_A, nnz_A, &nzval_A_dup, &rowind_A_dup, &colptr_A_dup);
        for (i = 0; i < m_A; ++i) {
            for (j = colptr_A[i]; j < colptr_A[i+1]; ++j) {
                nzval_A_dup[j] = nzval_A[j];
                rowind_A_dup[j] = rowind_A[j];
            }
            colptr_A_dup[i] = colptr_A[i];
        }
        colptr_A_dup[m_A] = colptr_A[m_A];

        fadi_col(options, m_A, nnz_A, nzval_A_dup, rowind_A_dup, colptr_A_dup, grid1, U1, ldu1, p1, q1, l1, tol, T1, r1, &rr1);

        if ( !(global_T1 = doubleMalloc_dist(m_A*rr1)) )
            ABORT("Malloc fails for global_T1[].");
        if ( !(newA = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newA[].");

        if (grid1->iam == 0) {
            printf("Grid 1 finishes fadi_col!\n");
            fflush(stdout);

            if ( !(tmpA = doubleMalloc_dist(m_A*rr1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(*T1, ldu1, global_T1, m_A, rr1, grid1);
        MPI_Bcast(global_T1, m_A*rr1, MPI_DOUBLE, 0, grid1->comm);

        if (grid1->iam == 0) {
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m_A, m_A, nnz_A, nzval_A_dup, rowind_A_dup, colptr_A_dup,
                SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rr1, one, &GA, global_T1, m_A, zero, tmpA, m_A);
            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rr1, &rr1, &m_A, &one, global_T1, &m_A, tmpA, &m_A, &zero, newA, &rr1);
        }
        MPI_Bcast(newA, rr1*rr1, MPI_DOUBLE, 0, grid1->comm);

        if ( !(newU2 = doubleMalloc_dist(rr1*ldlu2*r2)) )
            ABORT("Malloc fails for newU2[].");
        dgemm_("T", "N", &rr1, &ldnewu2, &m_A, &one, global_T1, &m_A, U2, &m_A, &zero, newU2, &rr1);

        *rank1 = rr1;

        dallocateA_dist(m_B, nnz_B, &nzval_B_neg, &rowind_B_neg, &colptr_B_neg);
        for (i = 0; i < m_B; ++i) {
            for (j = colptr_B[i]; j < colptr_B[i+1]; ++j) {
                nzval_B_neg[j] = -nzval_B[j];
                rowind_B_neg[j] = rowind_B[j];
            }
            colptr_B_neg[i] = colptr_B[i];
        }
        colptr_B_neg[m_B] = colptr_B[m_B];

        MPI_Barrier(grid1->comm);
    }

    for (j = 0; j < gr1; ++j) {
        root1 += grid_proc[j];
    }
    for (j = 0; j < gr2; ++j) {
        root2 += grid_proc[j];
    }

    if (grid1->iam == 0) {
        MPI_Send(rank1, 1, MPI_INT, root2, 0, MPI_COMM_WORLD);
    }
    else if (grid2->iam == 0) {
        MPI_Recv(rank1, 1, MPI_INT, root1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid2->iam != -1) {
        MPI_Bcast(rank1, 1, MPI_INT, 0, grid2->comm);
        rr1 = *rank1;
    
        dallocateA_dist(m_C, nnz_C, &nzval_C_neg, &rowind_C_neg, &colptr_C_neg);
        for (i = 0; i < m_C; ++i) {
            for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
                nzval_C_neg[j] = -nzval_C[j];
                rowind_C_neg[j] = rowind_C[j];
            }
            colptr_C_neg[i] = colptr_C[i];
        }
        colptr_C_neg[m_C] = colptr_C[m_C];

        MPI_Barrier(grid2->comm);
    }

    // printf("Process with id %d in A, %d in B, and %d in C gets here.\n", grid_A->iam, grid_B->iam, grid_C->iam);
    // fflush(stdout);

    if ((grid1->iam != -1) || (grid2->iam != -1)) {
        fadi_sp(options, rr1, newA, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg, 
            m_C, nnz_C, nzval_C_neg, rowind_C_neg, colptr_C_neg, grid1, grid2, 
            newU2, rr1*ldlu2, V2, ldv2, p2, q2, l2, tol, T2, T3, r2, rank2, la, ua, lb, ub, grid_proc, gr1, gr2);

        if (grid1->iam == 0) {
            printf("Grids finish fadi_sp!\n");
            fflush(stdout);
        }
    }
    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }
    
    if (grid1->iam == 0) {
        SUPERLU_FREE(tmpA);
    }
    if (grid1->iam != -1) {
        SUPERLU_FREE(newU2);
        SUPERLU_FREE(global_T1);
        SUPERLU_FREE(newA);
    }
}

void fadi_ttsvd_3d_2grids_1core(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double *T1, double **T2, double **T3, int r2, int rank1, int *rank2, int *grid_proc, int gr1, int gr2)
{
    SuperMatrix GA;
    double *newA, *tmpA, *newU2;
    double *global_T1;
    double *nzval_A_dup, *nzval_B_neg, *nzval_C_neg;
    int_t  *rowind_A_dup, *colptr_A_dup, *rowind_B_neg, *colptr_B_neg, *rowind_C_neg, *colptr_C_neg;
    int rr2, ldlu2 = ldu2/m_A;
    int ldnewu2 = ldlu2*r2;
    double one = 1.0, zero = 0.0;
    char     transpose[1];
    *transpose = 'N';
    int_t i, j;

    if (grid1->iam != -1) {
        dallocateA_dist(m_A, nnz_A, &nzval_A_dup, &rowind_A_dup, &colptr_A_dup);
        for (i = 0; i < m_A; ++i) {
            for (j = colptr_A[i]; j < colptr_A[i+1]; ++j) {
                nzval_A_dup[j] = nzval_A[j];
                rowind_A_dup[j] = rowind_A[j];
            }
            colptr_A_dup[i] = colptr_A[i];
        }
        colptr_A_dup[m_A] = colptr_A[m_A];

        if ( !(global_T1 = doubleMalloc_dist(m_A*rank1)) )
            ABORT("Malloc fails for global_T1[].");
        if ( !(newA = doubleMalloc_dist(rank1*rank1)) )
                ABORT("Malloc fails for newA[].");

        if (grid1->iam == 0) {
            if ( !(tmpA = doubleMalloc_dist(m_A*rank1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(T1, ldu1, global_T1, m_A, rank1, grid1);
        MPI_Bcast(global_T1, m_A*rank1, MPI_DOUBLE, 0, grid1->comm);

        if (grid1->iam == 0) {
            /* Create compressed column matrix for GA. */
            dCreate_CompCol_Matrix_dist(&GA, m_A, m_A, nnz_A, nzval_A_dup, rowind_A_dup, colptr_A_dup,
                SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rank1, one, &GA, global_T1, m_A, zero, tmpA, m_A);
            /* Destroy GA */
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rank1, &rank1, &m_A, &one, global_T1, &m_A, tmpA, &m_A, &zero, newA, &rank1);
        }
        MPI_Bcast(newA, rank1*rank1, MPI_DOUBLE, 0, grid1->comm);

        if ( !(newU2 = doubleMalloc_dist(rank1*ldlu2*r2)) )
            ABORT("Malloc fails for newU2[].");
        dgemm_("T", "N", &rank1, &ldnewu2, &m_A, &one, global_T1, &m_A, U2, &m_A, &zero, newU2, &rank1);

        dallocateA_dist(m_B, nnz_B, &nzval_B_neg, &rowind_B_neg, &colptr_B_neg);
        for (i = 0; i < m_B; ++i) {
            for (j = colptr_B[i]; j < colptr_B[i+1]; ++j) {
                nzval_B_neg[j] = -nzval_B[j];
                rowind_B_neg[j] = rowind_B[j];
            }
            colptr_B_neg[i] = colptr_B[i];
        }
        colptr_B_neg[m_B] = colptr_B[m_B];

        MPI_Barrier(grid1->comm);
    }

    if (grid2->iam != -1) {    
        dallocateA_dist(m_C, nnz_C, &nzval_C_neg, &rowind_C_neg, &colptr_C_neg);
        for (i = 0; i < m_C; ++i) {
            for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
                nzval_C_neg[j] = -nzval_C[j];
                rowind_C_neg[j] = rowind_C[j];
            }
            colptr_C_neg[i] = colptr_C[i];
        }
        colptr_C_neg[m_C] = colptr_C[m_C];

        // if (grid2->iam == 0) {
        //     printf("Get neg matrix.\n");
        //     fflush(stdout);
        // }

        MPI_Barrier(grid2->comm);
    }

    if ((grid1->iam != -1) || (grid2->iam != -1)) {
        fadi_sp(options, rank1, newA, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg, 
            m_C, nnz_C, nzval_C_neg, rowind_C_neg, colptr_C_neg, grid1, grid2, 
            newU2, rank1*ldlu2, V2, ldv2, p2, q2, l2, tol, T2, T3, r2, rank2, la, ua, lb, ub, grid_proc, gr1, gr2);

        if (grid1->iam == 0) {
            printf("Grids finish fadi_sp!\n");
            fflush(stdout);
        }
    }
    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }
    
    if (grid1->iam == 0) {
        SUPERLU_FREE(tmpA);
    }
    if (grid1->iam != -1) {
        SUPERLU_FREE(newU2);
        SUPERLU_FREE(global_T1);
        SUPERLU_FREE(newA);
    }
}

void fadi_ttsvd_3d_2grids_test(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, double *U1, int ldu1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc)
{
    double *nzval_A_dup;
    int_t  *rowind_A_dup, *colptr_A_dup;
    int ldlu2 = ldu2/m_A;
    int ldnewu2 = ldlu2*r2;
    double one = 1.0, zero = 0.0;
    char     transpose[1];
    *transpose = 'N';
    int_t i, j;

    if (grid1->iam != -1) {
        dallocateA_dist(m_A, nnz_A, &nzval_A_dup, &rowind_A_dup, &colptr_A_dup);
        for (i = 0; i < m_A; ++i) {
            for (j = colptr_A[i]; j < colptr_A[i+1]; ++j) {
                nzval_A_dup[j] = nzval_A[j];
                rowind_A_dup[j] = rowind_A[j];
            }
            colptr_A_dup[i] = colptr_A[i];
        }
        colptr_A_dup[m_A] = colptr_A[m_A];

        fadi_col(options, m_A, nnz_A, nzval_A_dup, rowind_A_dup, colptr_A_dup, grid1, U1, ldu1, p1, q1, l1, tol, T1, r1, rank1);

        MPI_Barrier(grid1->comm);
    }

    if (grid1->iam == 0) {
        MPI_Send(rank1, 1, MPI_INT, grid1->nprow*grid1->npcol, 0, MPI_COMM_WORLD);
    }
    else if (grid2->iam == 0) {
        MPI_Recv(rank1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid2->iam != -1) {
        MPI_Bcast(rank1, 1, MPI_INT, 0, grid2->comm);
    }

    fadi_ttsvd_3d_2grids_1core(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid1, grid2, ldu1, U2, ldu2, V2, ldv2,
        p2, q2, l2, tol, la, ua, lb, ub, *T1, T2, T3, r2, *rank1, rank2, grid_proc, 0, 1);
}

void fadi_ttsvd_3d_2grids_test2(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, double *U1, int ldu1, double *V1, int ldv1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub, double lc, double uc,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc)
{
    double *rT1, *rT2, *rT3;
    double *p1_neg, *q1_neg, *p2_neg, *q2_neg;
    int_t i, j, k;
    int rr1, rr2, ldlu2 = ldv1/m_C;

    if ( !(p1_neg = doubleMalloc_dist(l1)) )
        ABORT("Malloc fails for p1_neg[].");
    if ( !(q1_neg = doubleMalloc_dist(l1)) )
        ABORT("Malloc fails for q1_neg[].");
    for (j = 0; j < l1; ++j) {
        p1_neg[j] = -p1[j];
        q1_neg[j] = -q1[j];
    }
    if ( !(p2_neg = doubleMalloc_dist(l2)) )
        ABORT("Malloc fails for p2_neg[].");
    if ( !(q2_neg = doubleMalloc_dist(l2)) )
        ABORT("Malloc fails for q2_neg[].");
    for (j = 0; j < l2; ++j) {
        p2_neg[j] = -p2[j];
        q2_neg[j] = -q2[j];
    }

    // fadi_ttsvd_3d_2grids(options, m_C, nnz_C, nzval_C, rowind_C, colptr_C, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
    //     m_A, nnz_A, nzval_A, rowind_A, colptr_A, grid2, grid1, V2, ldv2, V1, ldv1, U1, ldu1, q2_neg, p2_neg, l2, q1_neg, p1_neg, l1, 
    //     tol, lc, uc, lb, ub, &rT1, &rT2, &rT3, r2, r1, rank2, rank1, grid_proc, 1, 0);

    if (grid2->iam != -1) {
        fadi_col(options, m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid2, V2, ldv2, q2_neg, p2_neg, l2, tol, &rT1, r2, rank2);
        MPI_Barrier(grid2->comm);
    }

    if (grid2->iam == 0) {
        MPI_Send(rank2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (grid1->iam == 0) {
        MPI_Recv(rank2, 1, MPI_INT, grid1->nprow*grid1->npcol, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid1->iam != -1) {
        MPI_Bcast(rank2, 1, MPI_INT, 0, grid1->comm);
    }

    fadi_ttsvd_3d_2grids_1core(options, m_C, nnz_C, nzval_C, rowind_C, colptr_C, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        m_A, nnz_A, nzval_A, rowind_A, colptr_A, grid2, grid1, ldv2, V1, ldv1, U1, ldu1,
        q1_neg, p1_neg, l1, tol, lc, uc, lb, ub, rT1, &rT2, &rT3, r1, *rank2, rank1, grid_proc, 1, 0);

    rr1 = *rank1;
    rr2 = *rank2;

    if (grid1->iam != -1) {
        if ( !(*T1 = doubleMalloc_dist(ldu1*rr1)) )
            ABORT("Malloc fails for *T1.");
        for (j = 0; j < rr1; ++j) {
            for (i = 0; i < ldu1; ++i) {
                (*T1)[j*ldu1+i] = rT3[i*rr1+j];
            }
        }

        SUPERLU_FREE(rT3);
    }
    if (grid2->iam != -1) {
        if ( !(*T2 = doubleMalloc_dist(rr1*ldlu2*rr2)) )
            ABORT("Malloc fails for *T2.");
        for (k = 0; k < ldlu2; ++k) {
            for (j = 0; j < rr2; ++j) {
                for (i = 0; i < rr1; ++i) {
                    (*T2)[j*ldlu2*rr1+k*rr1+i] = rT2[i*ldlu2*rr2+k*rr2+j];
                }
            }
        }

        if ( !(*T3 = doubleMalloc_dist(ldv2*rr2)) )
            ABORT("Malloc fails for *T3.");
        for (j = 0; j < ldv2; ++j) {
            for (i = 0; i < rr2; ++i) {
                (*T3)[j*rr2+i] = rT1[i*ldv2+j];
            }
        }

        SUPERLU_FREE(rT2);
        SUPERLU_FREE(rT1);
    }

    SUPERLU_FREE(p1_neg);
    SUPERLU_FREE(q1_neg);
    SUPERLU_FREE(p2_neg);
    SUPERLU_FREE(q2_neg);
}

void fadi_ttsvd_3d_2grids_rep(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid1, gridinfo_t *grid2, double *U1, int ldu1, double *V1, int ldv1, double *U2, int ldu2, double *V2, int ldv2,
    double *p1, double *q1, int_t l1, double *p2, double *q2, int_t l2, double tol, double la, double ua, double lb, double ub, double lc, double uc,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc, int rep, int *grid_main)
{
    if (rep == 0) {
        fadi_ttsvd_3d_2grids(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
            m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid1, grid2, U1, ldu1, U2, ldu2, V2, ldv2, p1, q1, l1, p2, q2, l2, 
            tol, la, ua, lb, ub, T1, T2, T3, r1, r2, rank1, rank2, grid_proc, 0, 1);
        *grid_main = 0;
        return;
    }

    int_t i, j, k, l;
    double *TTcores_old[3], *TTcores_new[3];
    int rr1, rr2;
    double *p1_neg, *q1_neg, *p2_neg, *q2_neg;
    int local2;

    if ( !(p1_neg = doubleMalloc_dist(l1)) )
        ABORT("Malloc fails for p1_neg[].");
    if ( !(q1_neg = doubleMalloc_dist(l1)) )
        ABORT("Malloc fails for q1_neg[].");
    for (j = 0; j < l1; ++j) {
        p1_neg[j] = -p1[j];
        q1_neg[j] = -q1[j];
    }

    fadi_ttsvd_3d_2grids(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid1, grid2, U1, ldu1, U2, ldu2, V2, ldv2, p1, q1, l1, p2, q2, l2, 
        tol, la, ua, lb, ub, &(TTcores_old[0]), &(TTcores_old[1]), &(TTcores_old[2]), r1, r2, &rr1, &rr2, grid_proc, 0, 1);

    if (grid1->iam == 0) {
        printf("Finish with basic solve.\n");
        fflush(stdout);
    }

    // if (grid1->iam != -1) {
    //     printf("After basic solve, T1 is\n");
    //     for (i = 0; i < ldu1; ++i) {
    //         for (j = 0; j < rr1; ++j) {
    //             printf("%f ", TTcores_old[0][j*ldu1+i]);
    //         }
    //         printf("\n");
    //     }
    //     printf("After basic solve, T2 is\n");
    //     for (i = 0; i < rr1; ++i) {
    //         for (j = 0; j < rr2; ++j) {
    //             printf("%f ", TTcores_old[0][j*ldu1+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (l = 0; l < rep; ++l) {
        if (l % 2 == 0) {
            if (grid2->iam != -1) {
                if ( !(TTcores_new[0] = doubleMalloc_dist(ldv2*rr2)) )
                    ABORT("Malloc fails for TTcores_new[0].");
                for (j = 0; j < rr2; ++j) {
                    for (i = 0; i < ldv2; ++i) {
                        TTcores_new[0][j*ldv2+i] = TTcores_old[2][i*rr2+j];
                    }
                }

                SUPERLU_FREE(TTcores_old[2]);
            }
            if (grid1->iam != -1) {
                SUPERLU_FREE(TTcores_old[0]);
                SUPERLU_FREE(TTcores_old[1]);
            }

            fadi_ttsvd_3d_2grids_1core(options, m_C, nnz_C, nzval_C, rowind_C, colptr_C, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
                m_A, nnz_A, nzval_A, rowind_A, colptr_A, grid2, grid1, ldv2, V1, ldv1, U1, ldu1,
                q1_neg, p1_neg, l1, tol, lc, uc, lb, ub, TTcores_new[0], &(TTcores_new[1]), &(TTcores_new[2]), r1, rr2, &rr1, grid_proc, 1, 0);

            if (grid2->iam == 0) {
                printf("Finish with rep number %d.\n", l+1);
                fflush(stdout);
            }
        }
        else {
            if (grid1->iam != -1) {
                if ( !(TTcores_old[0] = doubleMalloc_dist(ldu1*rr1)) )
                    ABORT("Malloc fails for TTcores_old[0].");
                for (j = 0; j < rr1; ++j) {
                    for (i = 0; i < ldu1; ++i) {
                        TTcores_old[0][j*ldu1+i] = TTcores_new[2][i*rr1+j];
                    }
                }

                SUPERLU_FREE(TTcores_new[2]);
            }
            if (grid2->iam != -1) {
                SUPERLU_FREE(TTcores_new[0]);
                SUPERLU_FREE(TTcores_new[1]);
            }

            fadi_ttsvd_3d_2grids_1core(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
                m_C, nnz_C, nzval_C, rowind_C, colptr_C, grid1, grid2, ldu1, U2, ldu2, V2, ldv2,
                p2, q2, l2, tol, la, ua, lb, ub, TTcores_old[0], &(TTcores_old[1]), &(TTcores_old[2]), r2, rr1, &rr2, grid_proc, 0, 1);

            if (grid1->iam == 0) {
                printf("Finish with rep number %d.\n", l+1);
                fflush(stdout);
            }
        }
    }

    *rank1 = rr1;
    *rank2 = rr2;
    if (rep % 2 == 0) {
        if (grid1->iam != -1) {
            local2 = ldu2 / m_A;

            if ( !(*T1 = doubleMalloc_dist(ldu1*rr1)) )
                ABORT("Malloc fails for *T1.");
            for (j = 0; j < rr1*ldu1; ++j) {
                (*T1)[j] = TTcores_old[0][j];
            }

            if ( !(*T2 = doubleMalloc_dist(rr1*local2*rr2)) )
                ABORT("Malloc fails for *T2.");
            for (j = 0; j < rr1*local2*rr2; ++j) {
                (*T2)[j] = TTcores_old[1][j];
            }

            SUPERLU_FREE(TTcores_old[0]);
            SUPERLU_FREE(TTcores_old[1]);
        }
        if (grid2->iam != -1) {
            if ( !(*T3 = doubleMalloc_dist(ldv2*rr2)) )
                ABORT("Malloc fails for *T3.");
            for (j = 0; j < rr2*ldv2; ++j) {
                (*T3)[j] = TTcores_old[2][j];
            }

            SUPERLU_FREE(TTcores_old[2]);
        }

        *grid_main = 0;
    }
    else {
        if (grid1->iam != -1) {
            if ( !(*T1 = doubleMalloc_dist(ldu1*rr1)) )
                ABORT("Malloc fails for *T1.");
            for (j = 0; j < rr1; ++j) {
                for (i = 0; i < ldu1; ++i) {
                    (*T1)[j*ldu1+i] = TTcores_new[2][i*rr1+j];
                }
            }

            SUPERLU_FREE(TTcores_new[2]);
        }
        if (grid2->iam != -1) {
            local2 = ldv1 / m_C;

            if ( !(*T2 = doubleMalloc_dist(rr1*local2*rr2)) )
                ABORT("Malloc fails for *T2.");
            for (k = 0; k < local2; ++k) {
                for (j = 0; j < rr2; ++j) {
                    for (i = 0; i < rr1; ++i) {
                        (*T2)[j*local2*rr1+k*rr1+i] = TTcores_new[1][i*local2*rr2+k*rr2+j];
                    }
                }
            }

            if ( !(*T3 = doubleMalloc_dist(ldv2*rr2)) )
                ABORT("Malloc fails for *T3.");
            for (j = 0; j < ldv2; ++j) {
                for (i = 0; i < rr2; ++i) {
                    (*T3)[j*rr2+i] = TTcores_new[0][i*ldv2+j];
                }
            }

            SUPERLU_FREE(TTcores_new[0]);
            SUPERLU_FREE(TTcores_new[1]);
        }

        *grid_main = 1;
    }

    SUPERLU_FREE(p1_neg);
    SUPERLU_FREE(q1_neg);
}

void fadi_ttsvd(superlu_dist_options_t options, int d, int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs,
    gridinfo_t *grid1, gridinfo_t *grid2, double **Us, double *V, int_t *locals, int_t *nrhss, double **ps, double **qs, int_t *ls, double tol,
    double *las, double *uas, double *lbs, double *ubs, double **TTcores, int *rs, int *grid_proc)
{
    SuperMatrix GA;
    double *tmpA;
    double *nzval_neg1, *nzval_neg2, *nzval_dup1;
    int_t  *rowind_neg1, *colptr_neg1, *rowind_neg2, *colptr_neg2, *rowind_dup1, *colptr_dup1;
    double **TTcores_global, **newA, **newU;
    int rr1;
    double one = 1.0, zero = 0.0;
    char transpose[1];
    *transpose = 'N';
    int_t i, j, k;

    if (d == 2) {
        double *D;

        if (grid2->iam != -1) {
            dallocateA_dist(ms[d-1], nnzs[d-1], &nzval_neg2, &rowind_neg2, &colptr_neg2);
            for (i = 0; i < ms[d-1]; ++i) {
                for (j = colptrs[d-1][i]; j < colptrs[d-1][i+1]; ++j) {
                    nzval_neg2[j] = -nzvals[d-1][j];
                    rowind_neg2[j] = rowinds[d-1][j];
                }
                colptr_neg2[i] = colptrs[d-1][i];
            }
            colptr_neg2[ms[d-1]] = colptrs[d-1][ms[d-1]];
        }

        fadi(options, ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0], ms[1], nnzs[1], nzval_neg2, rowind_neg2, colptr_neg2,
            grid1, grid2, Us[0], locals[0], V, locals[1], ps[0], qs[0], ls[0], tol, &(TTcores[0]), &D, &(TTcores[1]), nrhss[0], &rr1);
        rs[0] = rr1;

        if (grid1->iam > 0) {
            if ( !(D = doubleMalloc_dist(rr1)) )
                ABORT("Malloc fails for D[].");
        }
        if (grid1->iam != -1) {
            MPI_Bcast(D, rr1, MPI_DOUBLE, 0, grid1->comm);

            for (j = 0; j < rr1; ++j) {
                for (i = 0; i < locals[0]; ++i) {
                    TTcores[0][j*locals[0]+i] *= D[j];
                }
            }

            SUPERLU_FREE(D);
        }
        return;
    }
    else if (d == 3) {
        fadi_ttsvd_3d_2grids(options, ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0], ms[1], nnzs[1], nzvals[1], rowinds[1], colptrs[1],
            ms[2], nnzs[2], nzvals[2], rowinds[2], colptrs[2], grid1, grid2, Us[0], locals[0], Us[1], ms[0]*locals[1], 
            V, locals[2], ps[0], qs[0], ls[0], ps[1], qs[1], ls[1], tol, las[0], uas[0], lbs[0], ubs[0], &(TTcores[0]), &(TTcores[1]), 
            &(TTcores[2]), nrhss[0], nrhss[1], &(rs[0]), &(rs[1]), grid_proc, 0, 1);
        return;
    }

    newA = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));
    newU = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));

    if (grid1->iam != -1) {
        TTcores_global = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));

        dallocateA_dist(ms[0], nnzs[0], &nzval_dup1, &rowind_dup1, &colptr_dup1);
        for (i = 0; i < ms[0]; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzval_dup1[j] = nzvals[0][j];
                rowind_dup1[j] = rowinds[0][j];
            }
            colptr_dup1[i] = colptrs[0][i];
        }
        colptr_dup1[ms[0]] = colptrs[0][ms[0]];

        fadi_col(options, ms[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1, grid1, Us[0], locals[0], 
            ps[0], qs[0], ls[0], tol, &(TTcores[0]), nrhss[0], &rr1);
        // fadi_col(options, ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0], grid1, Us[0], locals[0], 
        //     ps[0], qs[0], ls[0], tol, &(TTcores[0]), nrhss[0], &rr1);
        rs[0] = rr1;

        if ( !(TTcores_global[0] = doubleMalloc_dist(ms[0]*rr1)) )
            ABORT("Malloc fails for TTcores_global[0][].");
        if ( !(newA[0] = doubleMalloc_dist(rr1*rr1)) )
            ABORT("Malloc fails for newA[0][].");

        if (grid1->iam == 0) {
            printf("Grid 1 finishes fadi_col for the first dimension with first TT rank %d!\n", rr1);
            fflush(stdout);

            if ( !(tmpA = doubleMalloc_dist(ms[0]*rr1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(TTcores[0], locals[0], TTcores_global[0], ms[0], rr1, grid1);

        if (grid1->iam == 0) {
            dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1,
                SLU_NC, SLU_D, SLU_GE);
            // dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0],
            //     SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rr1, one, &GA, TTcores_global[0], ms[0], zero, tmpA, ms[0]);
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rr1, &rr1, &(ms[0]), &one, TTcores_global[0], &(ms[0]), tmpA, &(ms[0]), &zero, newA[0], &rr1);
        }

        MPI_Bcast(TTcores_global[0], ms[0]*rr1, MPI_DOUBLE, 0, grid1->comm);
        MPI_Bcast(newA[0], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);
    }

    if (grid1->iam == 0) {
        MPI_Send(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD);
    }
    else if (grid2->iam == 0) {
        MPI_Recv(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid2->iam != -1) {
        MPI_Bcast(&rr1, 1, MPI_INT, 0, grid2->comm);
        rs[0] = rr1;

        MPI_Barrier(grid2->comm);
    }

    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }

    

    for (k = 1; k < d-2; ++k) {
        double *nzval_dup2;
        int_t *rowind_dup2, *colptr_dup2;

        if (grid1->iam != -1) {
            dmult_TTfADI_RHS(ms, rs, locals[k], k, Us[k], nrhss[k], TTcores_global, &(newU[k-1]));

            dallocateA_dist(ms[k], nnzs[k], &nzval_dup2, &rowind_dup2, &colptr_dup2);
            for (i = 0; i < ms[k]; ++i) {
                for (j = colptrs[k][i]; j < colptrs[k][i+1]; ++j) {
                    nzval_dup2[j] = nzvals[k][j];
                    rowind_dup2[j] = rowinds[k][j];
                }
                colptr_dup2[i] = colptrs[k][i];
            }
            colptr_dup2[ms[k]] = colptrs[k][ms[k]];

            fadi_col_adils(options, rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2, grid1, newU[k-1], locals[k],
                ps[k], qs[k], ls[k], las[k-1], uas[k-1], lbs[k-1], ubs[k-1], tol, &(TTcores[k]), nrhss[k], &rr1);
            // fadi_col_adils(options, rs[k-1], newA[k-1], ms[k], nnzs[k], nzvals[k], rowinds[k], colptrs[k], grid1, newU[k-1], locals[k],
            //     ps[k], qs[k], ls[k], las[k-1], uas[k-1], lbs[k-1], ubs[k-1], tol, &(TTcores[k]), nrhss[k], &rr1);
            rs[k] = rr1;

            if ( !(TTcores_global[k] = doubleMalloc_dist(rs[k-1]*ms[k]*rr1)) )
                ABORT("Malloc fails for TTcores_global[k][].");
            if ( !(newA[k] = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newA[k][].");

            if (grid1->iam == 0) {
                printf("Grid 1 finishes fadi_col for dimension %d with TT rank %d!\n", k+1, rr1);
                fflush(stdout);
            }

            dgather_X(TTcores[k], rs[k-1]*locals[k], TTcores_global[k], rs[k-1]*ms[k], rr1, grid1);
            if (grid1->iam == 0) {
                dmult_TTfADI_mat(rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2,
                    TTcores_global[k], rr1, newA[k]);
                // dmult_TTfADI_mat(rs[k-1], newA[k-1], ms[k], nnzs[k], nzvals[k], rowinds[k], colptrs[k],
                //     TTcores_global[k], rr1, newA[k]);
            }

            MPI_Bcast(TTcores_global[k], rs[k-1]*ms[k]*rr1, MPI_DOUBLE, 0, grid1->comm);
            MPI_Bcast(newA[k], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);
        }

        if (grid1->iam == 0) {
            MPI_Send(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD);
        }
        else if (grid2->iam == 0) {
            MPI_Recv(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (grid2->iam != -1) {
            MPI_Bcast(&rr1, 1, MPI_INT, 0, grid2->comm);
            rs[k] = rr1;

            MPI_Barrier(grid2->comm);
        }

        if (grid1->iam != -1) {
            MPI_Barrier(grid1->comm);
        }
    }

    if (grid1->iam != -1) {
        dallocateA_dist(ms[d-2], nnzs[d-2], &nzval_neg1, &rowind_neg1, &colptr_neg1);
        for (i = 0; i < ms[d-2]; ++i) {
            for (j = colptrs[d-2][i]; j < colptrs[d-2][i+1]; ++j) {
                nzval_neg1[j] = -nzvals[d-2][j];
                rowind_neg1[j] = rowinds[d-2][j];
            }
            colptr_neg1[i] = colptrs[d-2][i];
        }
        colptr_neg1[ms[d-2]] = colptrs[d-2][ms[d-2]];

        dmult_TTfADI_RHS(ms, rs, locals[d-2], d-2, Us[d-2], nrhss[d-2], TTcores_global, &(newU[d-3]));

        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        dallocateA_dist(ms[d-1], nnzs[d-1], &nzval_neg2, &rowind_neg2, &colptr_neg2);
        for (i = 0; i < ms[d-1]; ++i) {
            for (j = colptrs[d-1][i]; j < colptrs[d-1][i+1]; ++j) {
                nzval_neg2[j] = -nzvals[d-1][j];
                rowind_neg2[j] = rowinds[d-1][j];
            }
            colptr_neg2[i] = colptrs[d-1][i];
        }
        colptr_neg2[ms[d-1]] = colptrs[d-1][ms[d-1]];

        MPI_Barrier(grid2->comm);
    }
    if ((grid1->iam != -1) || (grid2->iam != -1)) {
        // if (grid2->iam != -1) {
        //     printf("Grid 2 proc %d starts final fadi_sp!\n", grid2->iam);
        //     fflush(stdout);
        // }

        fadi_sp(options, rs[d-3], newA[d-3], ms[d-2], nnzs[d-2], nzval_neg1, rowind_neg1, colptr_neg1, 
            ms[d-1], nnzs[d-1], nzval_neg2, rowind_neg2, colptr_neg2, grid1, grid2, 
            newU[d-3], rs[d-3]*locals[d-2], V, locals[d-1], ps[d-2], qs[d-2], ls[d-2], tol, &(TTcores[d-2]), &(TTcores[d-1]), 
            nrhss[d-2], &rr1, las[d-3], uas[d-3], lbs[d-3], ubs[d-3], grid_proc, 0, 1);
        rs[d-2] = rr1;

        if (grid1->iam == 0) {
            printf("Grid 1 and 2 finish final fadi_sp!\n");
            fflush(stdout);
        }
    }

    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }
    
    if (grid1->iam == 0) {
        SUPERLU_FREE(tmpA);
    }
    if (grid1->iam != -1) {
        for (j = 0; j < d-2; ++j) {
            SUPERLU_FREE(TTcores_global[j]);
            SUPERLU_FREE(newA[j]);
            SUPERLU_FREE(newU[j]);
        }
        SUPERLU_FREE(TTcores_global);
    }
    SUPERLU_FREE(newA);
    SUPERLU_FREE(newU);
}

void fadi_ttsvd_1core(superlu_dist_options_t options, int d, int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs,
    gridinfo_t *grid1, gridinfo_t *grid2, double **Us, double *V, int_t *locals, int_t *nrhss, double **ps, double **qs, int_t *ls, double tol,
    double *las, double *uas, double *lbs, double *ubs, double **TTcores, int *rs, int *grid_proc, int gr1, int gr2)
{
    SuperMatrix GA;
    double *tmpA;
    double *nzval_neg1, *nzval_neg2, *nzval_dup1;
    int_t  *rowind_neg1, *colptr_neg1, *rowind_neg2, *colptr_neg2, *rowind_dup1, *colptr_dup1;
    double **TTcores_global, **newA, **newU;
    int rr1;
    double one = 1.0, zero = 0.0;
    char transpose[1];
    *transpose = 'N';
    int_t i, j, k;
    int root1 = 0, root2 = 0;

    for (j = 0; j < gr1; ++j) {
        root1 += grid_proc[j];
    }
    for (j = 0; j < gr2; ++j) {
        root2 += grid_proc[j];
    }

    newA = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));
    newU = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));

    if (grid1->iam != -1) {
        dallocateA_dist(ms[0], nnzs[0], &nzval_dup1, &rowind_dup1, &colptr_dup1);
        for (i = 0; i < ms[0]; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzval_dup1[j] = nzvals[0][j];
                rowind_dup1[j] = rowinds[0][j];
            }
            colptr_dup1[i] = colptrs[0][i];
        }
        colptr_dup1[ms[0]] = colptrs[0][ms[0]];

        TTcores_global = (double **) SUPERLU_MALLOC((d-2)*sizeof(double*));

        rr1 = rs[0];

        if ( !(TTcores_global[0] = doubleMalloc_dist(ms[0]*rr1)) )
            ABORT("Malloc fails for TTcores_global[0][].");
        if ( !(newA[0] = doubleMalloc_dist(rr1*rr1)) )
            ABORT("Malloc fails for newA[0][].");

        if (grid1->iam == 0) {
            if ( !(tmpA = doubleMalloc_dist(ms[0]*rr1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(TTcores[0], locals[0], TTcores_global[0], ms[0], rr1, grid1);

        if (grid1->iam == 0) {
            dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1,
                SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rr1, one, &GA, TTcores_global[0], ms[0], zero, tmpA, ms[0]);
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rr1, &rr1, &(ms[0]), &one, TTcores_global[0], &(ms[0]), tmpA, &(ms[0]), &zero, newA[0], &rr1);
        }

        MPI_Bcast(TTcores_global[0], ms[0]*rr1, MPI_DOUBLE, 0, grid1->comm);
        MPI_Bcast(newA[0], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);

        MPI_Barrier(grid1->comm);
    }

    for (k = 1; k < d-2; ++k) {
        double *nzval_dup2;
        int_t *rowind_dup2, *colptr_dup2;

        if (grid1->iam != -1) {
            dmult_TTfADI_RHS(ms, rs, locals[k], k, Us[k], nrhss[k], TTcores_global, &(newU[k-1]));

            dallocateA_dist(ms[k], nnzs[k], &nzval_dup2, &rowind_dup2, &colptr_dup2);
            for (i = 0; i < ms[k]; ++i) {
                for (j = colptrs[k][i]; j < colptrs[k][i+1]; ++j) {
                    nzval_dup2[j] = nzvals[k][j];
                    rowind_dup2[j] = rowinds[k][j];
                }
                colptr_dup2[i] = colptrs[k][i];
            }
            colptr_dup2[ms[k]] = colptrs[k][ms[k]];

            fadi_col_adils(options, rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2, grid1, newU[k-1], locals[k],
                ps[k], qs[k], ls[k], las[k-1], uas[k-1], lbs[k-1], ubs[k-1], tol, &(TTcores[k]), nrhss[k], &rr1);
            rs[k] = rr1;

            if ( !(TTcores_global[k] = doubleMalloc_dist(rs[k-1]*ms[k]*rr1)) )
                ABORT("Malloc fails for TTcores_global[k][].");
            if ( !(newA[k] = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newA[k][].");

            if (grid1->iam == 0) {
                printf("Grid 1 finishes fadi_col for dimension %d with TT rank %d!\n", k+1, rr1);
                fflush(stdout);
            }

            dgather_X(TTcores[k], rs[k-1]*locals[k], TTcores_global[k], rs[k-1]*ms[k], rr1, grid1);
            if (grid1->iam == 0) {
                dmult_TTfADI_mat(rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2,
                    TTcores_global[k], rr1, newA[k]);
            }

            MPI_Bcast(TTcores_global[k], rs[k-1]*ms[k]*rr1, MPI_DOUBLE, 0, grid1->comm);
            MPI_Bcast(newA[k], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);
        }

        if (grid1->iam == 0) {
            MPI_Send(&rr1, 1, MPI_INT, root2, 0, MPI_COMM_WORLD);
        }
        else if (grid2->iam == 0) {
            MPI_Recv(&rr1, 1, MPI_INT, root1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (grid2->iam != -1) {
            MPI_Bcast(&rr1, 1, MPI_INT, 0, grid2->comm);
            rs[k] = rr1;

            MPI_Barrier(grid2->comm);
        }

        if (grid1->iam != -1) {
            MPI_Barrier(grid1->comm);
        }
    }

    if (grid1->iam != -1) {
        dallocateA_dist(ms[d-2], nnzs[d-2], &nzval_neg1, &rowind_neg1, &colptr_neg1);
        for (i = 0; i < ms[d-2]; ++i) {
            for (j = colptrs[d-2][i]; j < colptrs[d-2][i+1]; ++j) {
                nzval_neg1[j] = -nzvals[d-2][j];
                rowind_neg1[j] = rowinds[d-2][j];
            }
            colptr_neg1[i] = colptrs[d-2][i];
        }
        colptr_neg1[ms[d-2]] = colptrs[d-2][ms[d-2]];

        dmult_TTfADI_RHS(ms, rs, locals[d-2], d-2, Us[d-2], nrhss[d-2], TTcores_global, &(newU[d-3]));

        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        dallocateA_dist(ms[d-1], nnzs[d-1], &nzval_neg2, &rowind_neg2, &colptr_neg2);
        for (i = 0; i < ms[d-1]; ++i) {
            for (j = colptrs[d-1][i]; j < colptrs[d-1][i+1]; ++j) {
                nzval_neg2[j] = -nzvals[d-1][j];
                rowind_neg2[j] = rowinds[d-1][j];
            }
            colptr_neg2[i] = colptrs[d-1][i];
        }
        colptr_neg2[ms[d-1]] = colptrs[d-1][ms[d-1]];

        MPI_Barrier(grid2->comm);
    }
    if ((grid1->iam != -1) || (grid2->iam != -1)) {
        // if (grid2->iam != -1) {
        //     printf("Grid 2 proc %d starts final fadi_sp!\n", grid2->iam);
        //     fflush(stdout);
        // }

        fadi_sp(options, rs[d-3], newA[d-3], ms[d-2], nnzs[d-2], nzval_neg1, rowind_neg1, colptr_neg1, 
            ms[d-1], nnzs[d-1], nzval_neg2, rowind_neg2, colptr_neg2, grid1, grid2, 
            newU[d-3], rs[d-3]*locals[d-2], V, locals[d-1], ps[d-2], qs[d-2], ls[d-2], tol, &(TTcores[d-2]), &(TTcores[d-1]), 
            nrhss[d-2], &rr1, las[d-3], uas[d-3], lbs[d-3], ubs[d-3], grid_proc, gr1, gr2);
        rs[d-2] = rr1;

        if (grid1->iam == 0) {
            printf("Grid 1 and 2 finish final fadi_sp!\n");
            fflush(stdout);
        }
    }

    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }
    
    if (grid1->iam == 0) {
        SUPERLU_FREE(tmpA);
    }
    if (grid1->iam != -1) {
        for (j = 0; j < d-2; ++j) {
            SUPERLU_FREE(TTcores_global[j]);
            SUPERLU_FREE(newA[j]);
            SUPERLU_FREE(newU[j]);
        }
        SUPERLU_FREE(TTcores_global);
    }
    SUPERLU_FREE(newA);
    SUPERLU_FREE(newU);
}

void fadi_ttsvd_rep(superlu_dist_options_t options, int d, int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs,
    gridinfo_t *grid1, gridinfo_t *grid2, double **Us, double **Vs, int_t *locals1, int_t *locals2, int_t *nrhss, double **ps, double **qs, int_t *ls, double tol,
    double *las, double *uas, double *lbs, double *ubs, double **TTcores, int *rs, int *grid_proc, int rep, int *grid_main)
{
    if (rep == 0) {
        fadi_ttsvd(options, d, ms, nnzs, nzvals, rowinds, colptrs, grid1, grid2, Us, Vs[0], locals1, nrhss, ps, qs, ls, tol,
            las, uas, lbs, ubs, TTcores, rs, grid_proc);
        *grid_main = 0;

        return;
    }

    int_t i, j, k, l;
    double *TTcores_old[d], *TTcores_new[d];
    int rr1;
    double *ps_neg[d-1], *qs_neg[d-1];
    int_t ms_rev[d], nnzs_rev[d], nrhss_rev[d-1], ls_rev[d-1];
    int_t *rowinds_rev[d], *colptrs_rev[d];
    double las_rev[d-2], uas_rev[d-2], lbs_rev[d-2], ubs_rev[d-2];
    double *nzvals_rev[d];
    int rs_rev[d-1];

    for (j = 0; j < d; ++j) {
        ms_rev[j] = ms[d-1-j];
    }

    if (grid2->iam != -1) {
        for (j = 0; j < d-1; ++j) {
            nnzs_rev[j] = nnzs[d-1-j];

            dallocateA_dist(ms_rev[j], nnzs_rev[j], &nzvals_rev[j], &rowinds_rev[j], &colptrs_rev[j]);
            for (i = 0; i < ms_rev[j]; ++i) {
                for (k = colptrs[d-1-j][i]; k < colptrs[d-1-j][i+1]; ++k) {
                    nzvals_rev[j][k] = nzvals[d-1-j][k];
                    rowinds_rev[j][k] = rowinds[d-1-j][k];
                }
                colptrs_rev[j][i] = colptrs[d-1-j][i];
            }
            colptrs_rev[j][ms_rev[j]] = colptrs[d-1-j][ms[d-1-j]];
        }
    }
    if (grid1->iam != -1) {
        nnzs_rev[d-1] = nnzs[0];

        dallocateA_dist(ms_rev[d-1], nnzs_rev[d-1], &nzvals_rev[d-1], &rowinds_rev[d-1], &colptrs_rev[d-1]);
        for (i = 0; i < ms_rev[d-1]; ++i) {
            for (k = colptrs[0][i]; k < colptrs[0][i+1]; ++k) {
                nzvals_rev[d-1][k] = nzvals[0][k];
                rowinds_rev[d-1][k] = rowinds[0][k];
            }
            colptrs_rev[d-1][i] = colptrs[0][i];
        }
        colptrs_rev[d-1][ms_rev[d-1]] = colptrs[0][ms[0]];
    }

    for (j = 0; j < d-1; ++j) {
        nrhss_rev[j] = nrhss[d-2-j];
        ls_rev[j] = ls[d-2-j];

        if ( !(ps_neg[j] = doubleMalloc_dist(ls_rev[j])) )
            ABORT("Malloc fails for ps_neg[j][].");
        if ( !(qs_neg[j] = doubleMalloc_dist(ls_rev[j])) )
            ABORT("Malloc fails for qs_neg[j][].");
        for (i = 0; i < ls_rev[j]; ++i) {
            ps_neg[j][i] = -ps[d-2-j][i];
            qs_neg[j][i] = -qs[d-2-j][i];
        }
    }

    if ((grid1->iam == 0) || (grid2->iam == 0)) {
        for (j = 0; j < d-2; ++j) {
            las_rev[j] = las[d-2+j];
            uas_rev[j] = uas[d-2+j];
            lbs_rev[j] = lbs[d-2+j];
            ubs_rev[j] = ubs[d-2+j];
        }
    }

    fadi_ttsvd(options, d, ms, nnzs, nzvals, rowinds, colptrs, grid1, grid2, Us, Vs[0], locals1, nrhss, ps, qs, ls, tol,
            las, uas, lbs, ubs, TTcores_old, rs, grid_proc);
    rs_rev[0] = rs[d-2];

    if (grid1->iam == 0) {
        printf("Finish with basic solve.\n");
        fflush(stdout);
    }

    for (l = 0; l < rep; ++l) {
        if (l % 2 == 0) {
            if (grid2->iam != -1) {
                if ( !(TTcores_new[0] = doubleMalloc_dist(locals2[0]*rs_rev[0])) )
                    ABORT("Malloc fails for TTcores_new[0].");
                for (j = 0; j < rs_rev[0]; ++j) {
                    for (i = 0; i < locals2[0]; ++i) {
                        TTcores_new[0][j*locals2[0]+i] = TTcores_old[d-1][i*rs[d-2]+j];
                    }
                }

                SUPERLU_FREE(TTcores_old[d-1]);
            }
            if (grid1->iam != -1) {
                for (j = 0; j < d-1; ++j) {
                    SUPERLU_FREE(TTcores_old[j]);
                }
            }

            fadi_ttsvd_1core(options, d, ms_rev, nnzs_rev, nzvals_rev, rowinds_rev, colptrs_rev, grid2, grid1, Vs, Us[0], locals2, 
                nrhss_rev, qs_neg, ps_neg, ls_rev, tol, las_rev, uas_rev, lbs_rev, ubs_rev, TTcores_new, rs_rev, grid_proc, 1, 0);
            rs[0] = rs_rev[d-2];

            if (grid2->iam == 0) {
                printf("Finish with rep number %d.\n", l+1);
                fflush(stdout);
            }
        }
        else {
            if (grid1->iam != -1) {
                if ( !(TTcores_old[0] = doubleMalloc_dist(locals1[0]*rs[0])) )
                    ABORT("Malloc fails for TTcores_old[0].");
                for (j = 0; j < rs[0]; ++j) {
                    for (i = 0; i < locals1[0]; ++i) {
                        TTcores_old[0][j*locals1[0]+i] = TTcores_new[d-1][i*rs_rev[d-2]+j];
                    }
                }

                SUPERLU_FREE(TTcores_new[d-1]);
            }
            if (grid2->iam != -1) {
                for (j = 0; j < d-1; ++j) {
                    SUPERLU_FREE(TTcores_new[j]);
                }
            }

            fadi_ttsvd_1core(options, d, ms, nnzs, nzvals, rowinds, colptrs, grid1, grid2, Us, Vs[0], locals1, 
                nrhss, ps, qs, ls, tol, las, uas, lbs, ubs, TTcores_old, rs, grid_proc, 0, 1);
            rs_rev[0] = rs[d-2];

            if (grid1->iam == 0) {
                printf("Finish with rep number %d.\n", l+1);
                fflush(stdout);
            }
        }
    }

    if (rep % 2 == 0) {
        if (grid1->iam != -1) {
            for (j = 0; j < d-1; ++j) {
                l = j == 0 ? locals1[0]*rs[0] : rs[j-1]*locals1[j]*rs[j];
                if ( !(TTcores[j] = doubleMalloc_dist(l)) )
                    ABORT("Malloc fails for TTcores[j].");
                for (i = 0; i < l; ++i) {
                    TTcores[j][i] = TTcores_old[j][i];
                }

                SUPERLU_FREE(TTcores_old[j]);
            }
        }
        if (grid2->iam != -1) {
            if ( !(TTcores[d-1] = doubleMalloc_dist(locals1[d-1]*rs[d-2])) )
                ABORT("Malloc fails for TTcores[d-1].");
            for (i = 0; i < locals1[d-1]*rs[d-2]; ++i) {
                TTcores[d-1][i] = TTcores_old[d-1][i];
            }

            SUPERLU_FREE(TTcores_old[d-1]);
        }
        *grid_main = 0;
    }
    else {
        // printf("Reversed rank computed for proc %d in grid1 and %d in grid2 is ", grid1->iam, grid2->iam);
        // for (j = 0; j < d-1; ++j) {
        //     printf("%d ", rs_rev[j]);
        // }
        // printf("\n");
        // fflush(stdout);

        dtranspose_TTcores(rs_rev, rs, locals2, d, TTcores_new, TTcores, grid1, grid2);
        *grid_main = 1;
    }

    for (j = 0; j < d-1; ++j) {
        SUPERLU_FREE(ps_neg[j]);
        SUPERLU_FREE(qs_neg[j]);
    }
}

void fadi_ttsvd_2way(superlu_dist_options_t options, int d, int_t *ms, int_t *nnzs, double **nzvals, int_t **rowinds, int_t **colptrs,
    gridinfo_t *grid1, gridinfo_t *grid2, double **Us, double **Vs, int_t *locals, int_t *nrhss, double **ps, double **qs, int_t *ls, double tol,
    double *las, double *uas, double *lbs, double *ubs, double **TTcores, int *rs, int *grid_proc)
{
    if (d == 3) {
        fadi_ttsvd_3d_2grids(options, ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0], ms[1], nnzs[1], nzvals[1], rowinds[1], colptrs[1],
            ms[2], nnzs[2], nzvals[2], rowinds[2], colptrs[2], grid1, grid2, Us[0], locals[0], Us[1], ms[0]*locals[1], 
            Vs[0], locals[2], ps[0], qs[0], ls[0], ps[1], qs[1], ls[1], tol, las[0], uas[0], lbs[0], ubs[0], &(TTcores[0]), &(TTcores[1]), 
            &(TTcores[2]), nrhss[0], nrhss[1], &(rs[0]), &(rs[1]), grid_proc, 0, 1);
        return;
    }

    SuperMatrix GA, GB;
    double *tmpA, *tmpB;
    double *nzval_neg1, *nzval_neg2, *nzval_dup1;
    int_t  *rowind_neg1, *colptr_neg1, *rowind_neg2, *colptr_neg2, *rowind_dup1, *colptr_dup1;
    double **TTcores_global, **TTcores_rev_global, **newA, **newB, **newU, **newV;
    double one = 1.0, zero = 0.0;
    char transpose[1];
    *transpose = 'N';
    int_t i, j, k, l;
    int rr1, deal;

    deal = d/2;
    if ((grid1->iam != -1) && (d % 2 == 1)) {
        deal++;
    }
    double *TTcores_rev[deal];
    int_t ms_rev[d];
    int rs_rev[d-1];

    for (j = 0; j < d; ++j) {
        ms_rev[j] = ms[d-1-j];
    }

    newA = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));
    newB = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));
    newU = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));
    newV = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));
    TTcores_global = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));
    TTcores_rev_global = (double **) SUPERLU_MALLOC((deal-1)*sizeof(double*));

    if (grid1->iam != -1) {
        dallocateA_dist(ms[0], nnzs[0], &nzval_dup1, &rowind_dup1, &colptr_dup1);
        for (i = 0; i < ms[0]; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzval_dup1[j] = nzvals[0][j];
                rowind_dup1[j] = rowinds[0][j];
            }
            colptr_dup1[i] = colptrs[0][i];
        }
        colptr_dup1[ms[0]] = colptrs[0][ms[0]];

        fadi_col(options, ms[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1, grid1, Us[0], locals[0], 
            ps[0], qs[0], ls[0], tol, &(TTcores[0]), nrhss[0], &rr1);
        rs[0] = rr1;
        rs_rev[d-2] = rr1;

        if ( !(TTcores_global[0] = doubleMalloc_dist(ms[0]*rr1)) )
            ABORT("Malloc fails for TTcores_global[0][].");
        if ( !(newA[0] = doubleMalloc_dist(rr1*rr1)) )
            ABORT("Malloc fails for newA[0][].");

        if (grid1->iam == 0) {
            printf("Grid 1 finishes fadi_col for the first dimension with first TT rank %d!\n", rr1);
            fflush(stdout);

            if ( !(tmpA = doubleMalloc_dist(ms[0]*rr1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(TTcores[0], locals[0], TTcores_global[0], ms[0], rr1, grid1);

        if (grid1->iam == 0) {
            dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1,
                SLU_NC, SLU_D, SLU_GE);
            // dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0],
            //     SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rr1, one, &GA, TTcores_global[0], ms[0], zero, tmpA, ms[0]);
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rr1, &rr1, &(ms[0]), &one, TTcores_global[0], &(ms[0]), tmpA, &(ms[0]), &zero, newA[0], &rr1);
        }

        MPI_Bcast(TTcores_global[0], ms[0]*rr1, MPI_DOUBLE, 0, grid1->comm);
        MPI_Bcast(newA[0], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);

        // if (grid1->iam == 0) {
        //     printf("Grid 1 finishes entire first iteration!\n");
        //     fflush(stdout);
        // }
    }
    if (grid1->iam == 0) {
        MPI_Send(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD);
    }
    else if (grid2->iam == 0) {
        MPI_Recv(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid2->iam != -1) {
        MPI_Bcast(&rr1, 1, MPI_INT, 0, grid2->comm);
        rs[0] = rr1;
        rs_rev[d-2] = rr1;

        dallocateA_dist(ms_rev[0], nnzs[0], &nzval_dup1, &rowind_dup1, &colptr_dup1);
        for (i = 0; i < ms_rev[0]; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzval_dup1[j] = nzvals[0][j];
                rowind_dup1[j] = rowinds[0][j];
            }
            colptr_dup1[i] = colptrs[0][i];
        }
        colptr_dup1[ms_rev[0]] = colptrs[0][ms_rev[0]];

        fadi_col(options, ms_rev[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1, grid2, Vs[0], locals[0], 
            ps[0], qs[0], ls[0], tol, &(TTcores_rev[0]), nrhss[0], &rr1);
        rs[d-2] = rr1;
        rs_rev[0] = rr1;

        if ( !(TTcores[d-1] = doubleMalloc_dist(ms_rev[0]*rr1)) )
            ABORT("Malloc fails for TTcores[d-1][].");
        for (j = 0; j < locals[0]; ++j) {
            for (i = 0; i < rr1; ++i) {
                TTcores[d-1][j*rr1+i] = TTcores_rev[0][i*locals[0]+j];
            }
        }

        if ( !(TTcores_rev_global[0] = doubleMalloc_dist(ms_rev[0]*rr1)) )
            ABORT("Malloc fails for TTcores_global[0][].");
        if ( !(newB[0] = doubleMalloc_dist(rr1*rr1)) )
            ABORT("Malloc fails for newB[0][].");

        if (grid2->iam == 0) {
            printf("Grid 2 finishes fadi_col for the last dimension with last TT rank %d!\n", rr1);
            fflush(stdout);

            if ( !(tmpA = doubleMalloc_dist(ms_rev[0]*rr1)) )
                ABORT("Malloc fails for tmpA[].");
        }
        dgather_X(TTcores_rev[0], locals[0], TTcores_rev_global[0], ms_rev[0], rr1, grid2);

        if (grid2->iam == 0) {
            dCreate_CompCol_Matrix_dist(&GA, ms_rev[0], ms_rev[0], nnzs[0], nzval_dup1, rowind_dup1, colptr_dup1,
                SLU_NC, SLU_D, SLU_GE);
            // dCreate_CompCol_Matrix_dist(&GA, ms[0], ms[0], nnzs[0], nzvals[0], rowinds[0], colptrs[0],
            //     SLU_NC, SLU_D, SLU_GE);
            sp_dgemm_dist(transpose, rr1, one, &GA, TTcores_rev_global[0], ms_rev[0], zero, tmpA, ms_rev[0]);
            Destroy_CompCol_Matrix_dist(&GA);

            dgemm_("T", "N", &rr1, &rr1, &(ms_rev[0]), &one, TTcores_rev_global[0], &(ms_rev[0]), tmpA, &(ms_rev[0]), &zero, newB[0], &rr1);
        }

        MPI_Bcast(TTcores_rev_global[0], ms_rev[0]*rr1, MPI_DOUBLE, 0, grid2->comm);
        MPI_Bcast(newB[0], rr1*rr1, MPI_DOUBLE, 0, grid2->comm);
    }  
    if (grid2->iam == 0) {
        MPI_Send(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (grid1->iam == 0) {
        MPI_Recv(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (grid1->iam != -1) {
        MPI_Bcast(&rr1, 1, MPI_INT, 0, grid1->comm);
        rs[d-2] = rr1;
        rs_rev[0] = rr1;

        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }

    

    for (k = 1; k < deal-1; ++k) {
        double *nzval_dup2;
        int_t *rowind_dup2, *colptr_dup2;

        if (grid1->iam != -1) {
            dmult_TTfADI_RHS(ms, rs, locals[k], k, Us[k], nrhss[k], TTcores_global, &(newU[k-1]));

            dallocateA_dist(ms[k], nnzs[k], &nzval_dup2, &rowind_dup2, &colptr_dup2);
            for (i = 0; i < ms[k]; ++i) {
                for (j = colptrs[k][i]; j < colptrs[k][i+1]; ++j) {
                    nzval_dup2[j] = nzvals[k][j];
                    rowind_dup2[j] = rowinds[k][j];
                }
                colptr_dup2[i] = colptrs[k][i];
            }
            colptr_dup2[ms[k]] = colptrs[k][ms[k]];

            fadi_col_adils(options, rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2, grid1, newU[k-1], locals[k],
                ps[k], qs[k], ls[k], las[k-1], uas[k-1], lbs[k-1], ubs[k-1], tol, &(TTcores[k]), nrhss[k], &rr1);
            rs[k] = rr1;
            rs_rev[d-2-k] = rr1;

            if ( !(TTcores_global[k] = doubleMalloc_dist(rs[k-1]*ms[k]*rr1)) )
                ABORT("Malloc fails for TTcores_global[k][].");
            if ( !(newA[k] = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newA[k][].");

            if (grid1->iam == 0) {
                printf("Grid 1 finishes fadi_col for dimension %d with TT rank %d!\n", k+1, rr1);
                fflush(stdout);
            }

            dgather_X(TTcores[k], rs[k-1]*locals[k], TTcores_global[k], rs[k-1]*ms[k], rr1, grid1);
            if (grid1->iam == 0) {
                dmult_TTfADI_mat(rs[k-1], newA[k-1], ms[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2,
                    TTcores_global[k], rr1, newA[k]);
            }

            MPI_Bcast(TTcores_global[k], rs[k-1]*ms[k]*rr1, MPI_DOUBLE, 0, grid1->comm);
            MPI_Bcast(newA[k], rr1*rr1, MPI_DOUBLE, 0, grid1->comm);
        }
        if (grid1->iam == 0) {
            MPI_Send(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD);
        }
        else if (grid2->iam == 0) {
            MPI_Recv(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (grid2->iam != -1) {
            MPI_Bcast(&rr1, 1, MPI_INT, 0, grid2->comm);
            rs[k] = rr1;
            rs_rev[d-2-k] = rr1;

            dmult_TTfADI_RHS_alt(ms_rev, rs_rev, locals[k], k, Vs[k], nrhss[k], TTcores_rev_global, &(newV[k-1]));

            dallocateA_dist(ms_rev[k], nnzs[k], &nzval_dup2, &rowind_dup2, &colptr_dup2);
            for (i = 0; i < ms_rev[k]; ++i) {
                for (j = colptrs[k][i]; j < colptrs[k][i+1]; ++j) {
                    nzval_dup2[j] = nzvals[k][j];
                    rowind_dup2[j] = rowinds[k][j];
                }
                colptr_dup2[i] = colptrs[k][i];
            }
            colptr_dup2[ms_rev[k]] = colptrs[k][ms_rev[k]];

            fadi_col_adils(options, rs_rev[k-1], newB[k-1], ms_rev[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2, grid2, newV[k-1], locals[k],
                ps[k], qs[k], ls[k], las[k-1], uas[k-1], lbs[k-1], ubs[k-1], tol, &(TTcores_rev[k]), nrhss[k], &rr1);
            rs_rev[k] = rr1;
            rs[d-2-k] = rr1;

            if ( !(TTcores[d-1-k] = doubleMalloc_dist(rs_rev[k-1]*locals[k]*rr1)) )
                ABORT("Malloc fails for TTcores_global[k][].");
            for (l = 0; l < locals[k]; ++l) {
                for (j = 0; j < rs_rev[k-1]; ++j) {
                    for (i = 0; i < rr1; ++i) {
                        TTcores[d-1-k][j*locals[k]*rr1+l*rr1+i]
                            = TTcores_rev[k][i*locals[k]*rs_rev[k-1]+l*rs_rev[k-1]+j];
                    }
                }
            }

            if ( !(TTcores_rev_global[k] = doubleMalloc_dist(rs_rev[k-1]*ms_rev[k]*rr1)) )
                ABORT("Malloc fails for TTcores_global[k][].");
            if ( !(newB[k] = doubleMalloc_dist(rr1*rr1)) )
                ABORT("Malloc fails for newB[k][].");

            if (grid2->iam == 0) {
                printf("Grid 2 finishes fadi_col for dimension %d with TT rank %d!\n", d-k, rr1);
                fflush(stdout);
            }

            dgather_X(TTcores_rev[k], rs[k-1]*locals[k], TTcores_rev_global[k], rs_rev[k-1]*ms[k], rr1, grid2);
            if (grid2->iam == 0) {
                dmult_TTfADI_mat(rs_rev[k-1], newB[k-1], ms_rev[k], nnzs[k], nzval_dup2, rowind_dup2, colptr_dup2,
                    TTcores_rev_global[k], rr1, newB[k]);
            }

            MPI_Bcast(TTcores_rev_global[k], rs_rev[k-1]*ms[k]*rr1, MPI_DOUBLE, 0, grid2->comm);
            MPI_Bcast(newB[k], rr1*rr1, MPI_DOUBLE, 0, grid2->comm);
        }
        if (grid2->iam == 0) {
            MPI_Send(&rr1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (grid1->iam == 0) {
            MPI_Recv(&rr1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (grid1->iam != -1) {
            MPI_Bcast(&rr1, 1, MPI_INT, 0, grid1->comm);
            rs_rev[k] = rr1;
            rs[d-2-k] = rr1;
            MPI_Barrier(grid1->comm);
        }
        if (grid2->iam != -1) {
            MPI_Barrier(grid2->comm);
        }
    }

    if (grid1->iam != -1) {
        dallocateA_dist(ms[deal-1], nnzs[deal-1], &nzval_neg1, &rowind_neg1, &colptr_neg1);
        for (i = 0; i < ms[deal-1]; ++i) {
            for (j = colptrs[deal-1][i]; j < colptrs[deal-1][i+1]; ++j) {
                nzval_neg1[j] = -nzvals[deal-1][j];
                rowind_neg1[j] = rowinds[deal-1][j];
            }
            colptr_neg1[i] = colptrs[deal-1][i];
        }
        colptr_neg1[ms[deal-1]] = colptrs[deal-1][ms[deal-1]];

        dmult_TTfADI_RHS(ms, rs, locals[deal-1], deal-1, Us[deal-1], nrhss[deal-1], TTcores_global, &(newU[deal-2]));

        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        dallocateA_dist(ms_rev[deal-1], nnzs[deal-1], &nzval_neg2, &rowind_neg2, &colptr_neg2);
        for (i = 0; i < ms_rev[deal-1]; ++i) {
            for (j = colptrs[deal-1][i]; j < colptrs[deal-1][i+1]; ++j) {
                nzval_neg2[j] = nzvals[deal-1][j];
                rowind_neg2[j] = rowinds[deal-1][j];
            }
            colptr_neg2[i] = colptrs[deal-1][i];
        }
        colptr_neg2[ms_rev[deal-1]] = colptrs[deal-1][ms_rev[deal-1]];

        dmult_TTfADI_RHS_alt(ms_rev, rs_rev, locals[deal-1], deal-1, Vs[deal-1], nrhss[deal-1], TTcores_rev_global, &(newV[deal-2]));

        for (j = 0; j < rs_rev[deal-2]*rs_rev[deal-2]; ++j) {
            newB[deal-2][j] = -newB[deal-2][j];
        }

        MPI_Barrier(grid2->comm);
    }
    if ((grid1->iam != -1) || (grid2->iam != -1)) {
        fadi_sp_2sided(options, rs[deal-2], newA[deal-2], ms[deal-1], nnzs[deal-1], nzval_neg1, rowind_neg1, colptr_neg1,
            ms_rev[deal-1], nnzs[deal-1], nzval_neg2, rowind_neg2, colptr_neg2, rs_rev[deal-2], newB[deal-2], grid1, grid2, 
            newU[deal-2], rs[deal-2]*locals[deal-1], newV[deal-2], rs_rev[deal-2]*locals[deal-1], ps[deal-1], qs[deal-1], ls[deal-1], 
            tol, &(TTcores[deal-1]), &(TTcores[deal]), nrhss[deal-1], &rr1, las[deal-2], uas[deal-2], lbs[deal-2], ubs[deal-2], 
            las[deal-2], uas[deal-2], lbs[deal-2], ubs[deal-2], grid_proc, 0, 1);
        rs[deal-1] = rr1;

        if (grid1->iam == 0) {
            printf("Grid 1 and 2 finish final fadi_sp!\n");
            fflush(stdout);
        }
    }

    if (grid1->iam != -1) {
        MPI_Barrier(grid1->comm);
    }
    if (grid2->iam != -1) {
        MPI_Barrier(grid2->comm);
    }
    
    if ((grid1->iam == 0) || (grid2->iam == 0)) {
        SUPERLU_FREE(tmpA);
    }
    if (grid1->iam != -1) {
        for (j = 0; j < deal-1; ++j) {
            SUPERLU_FREE(TTcores_global[j]);
            SUPERLU_FREE(newA[j]);
            SUPERLU_FREE(newU[j]);
        }
    }
    if (grid2->iam != -1) {
        for (j = 0; j < deal-1; ++j) {
            SUPERLU_FREE(TTcores_rev_global[j]);
            SUPERLU_FREE(newB[j]);
            SUPERLU_FREE(newV[j]);
        }
    }
    SUPERLU_FREE(TTcores_global);
    SUPERLU_FREE(newA);
    SUPERLU_FREE(newU);
    SUPERLU_FREE(TTcores_rev_global);
    SUPERLU_FREE(newB);
    SUPERLU_FREE(newV);
}

void fadi_dimPara_ttsvd_3d(superlu_dist_options_t options, int_t m_A, int_t nnz_A, double *nzval_A, int_t *rowind_A, int_t *colptr_A,
    int_t m_B, int_t nnz_B, double *nzval_B, int_t *rowind_B, int_t *colptr_B,
    int_t m_C, int_t nnz_C, double *nzval_C, int_t *rowind_C, int_t *colptr_C,
    gridinfo_t *grid_A, gridinfo_t *grid_B, gridinfo_t *grid_C, double *U1, int ldu1, 
    double *U2, int ldu2, double *U2T, int ldu2t, double *V2, int ldv2,
    double *p, double *q, int_t l, double tol, double la, double ua, double lb, double ub,
    double **T1, double **T2, double **T3, int r1, int r2, int *rank1, int *rank2, int *grid_proc)
{
    SuperMatrix A, C;
    double *nzval_B_neg, *nzval_C_neg;
    int_t  *rowind_B_neg, *colptr_B_neg, *rowind_C_neg, *colptr_C_neg;
    int rr2, tmpr2, ovsamp;
    double one = 1.0, zero = 0.0;
    char     transpose[1];
    *transpose = 'N';
    double *rhs_A, *rhs_B, *rhs_BT, *rhs_C;
    double *localZ, *localW_T, *localD, *localY;
    double *iterZ, *iterW_comb, *iterW_comb_tmp, *iterW, *iterW_T, *iterY;
    double *global_T1, *global_T1_onB;
    int info;
    int_t i, j, k, m;
    double *berr_A, *berr_C;
    SuperLUStat_t stat_A, stat_C;
    dScalePermstruct_t ScalePermstruct_A, ScalePermstruct_C;
    dLUstruct_t LUstruct_A, LUstruct_C;
    dSOLVEstruct_t SOLVEstruct_A, SOLVEstruct_C;
    int iam_A = grid_A->iam, iam_B = grid_B->iam, iam_C = grid_C->iam;

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (iam_A != -1) {
        if ( !(berr_A = doubleMalloc_dist(r1)) )
            ABORT("Malloc fails for berr_A[].");
        if ( !(rhs_A = doubleMalloc_dist(ldu1*r1)) )
            ABORT("Malloc fails for rhs_A[].");
        if ( !(localZ = doubleMalloc_dist(ldu1*r1*l)) )
            ABORT("Malloc fails for localZ[].");
        if ( !(iterZ = doubleMalloc_dist(ldu1*r1)) )
            ABORT("Malloc fails for iterZ[].");
        if ( !(rhs_B = doubleMalloc_dist(ldu2*m_B*r2)) )
            ABORT("Malloc fails for rhs_B[].");
        if ( !(iterW = doubleMalloc_dist(ldu2*m_B*r2)) )
            ABORT("Malloc fails for iterW[].");
        if ( !(iterW_comb = doubleMalloc_dist(m_A*m_B*r2)) )
            ABORT("Malloc fails for iterW_comb[].");
    }
    if (iam_A == 0) {
        if ( !(iterW_comb_tmp = doubleMalloc_dist(m_A*m_B*r2)) )
            ABORT("Malloc fails for iterW_comb_tmp[].");
    }

    if (iam_B != -1) {
        if ( !(rhs_BT = doubleMalloc_dist(ldu2t*m_A*r2)) )
            ABORT("Malloc fails for rhs_BT[].");
        if ( !(localW_T = doubleMalloc_dist(ldu2t*m_A*r2*l)) )
            ABORT("Malloc fails for localW_T[].");
        if ( !(iterW_T = doubleMalloc_dist(ldu2t*m_A*r2)) )
            ABORT("Malloc fails for iterW_T[].");
        if ( !(localD = doubleMalloc_dist(r2*l)) )
            ABORT("Malloc fails for localD[].");

        dallocateA_dist(m_B, nnz_B, &nzval_B_neg, &rowind_B_neg, &colptr_B_neg);
        for (i = 0; i < m_B; ++i) {
            for (j = colptr_B[i]; j < colptr_B[i+1]; ++j) {
                nzval_B_neg[j] = -nzval_B[j];
                rowind_B_neg[j] = rowind_B[j];
            }
            colptr_B_neg[i] = colptr_B[i];
        }
        colptr_B_neg[m_B] = colptr_B[m_B];
    }

    if (iam_C != -1) {
        if ( !(berr_C = doubleMalloc_dist(r2)) )
            ABORT("Malloc fails for berr_C[].");
        if ( !(rhs_C = doubleMalloc_dist(ldv2*r2)) )
            ABORT("Malloc fails for rhs_C[].");
        if ( !(localY = doubleMalloc_dist(ldv2*r2*l)) )
            ABORT("Malloc fails for localY[].");
        if ( !(iterY = doubleMalloc_dist(ldv2*r2)) )
            ABORT("Malloc fails for iterY[].");

        dallocateA_dist(m_C, nnz_C, &nzval_C_neg, &rowind_C_neg, &colptr_C_neg);
        for (i = 0; i < m_C; ++i) {
            for (j = colptr_C[i]; j < colptr_C[i+1]; ++j) {
                nzval_C_neg[j] = -nzval_C[j];
                rowind_C_neg[j] = rowind_C[j];
            }
            colptr_C_neg[i] = colptr_C[i];
        }
        colptr_C_neg[m_C] = colptr_C[m_C];
    }

    if (iam_A != -1) {
        dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[0]);

        for (j = 0; j < r1; ++j) {
            for (i = 0; i < ldu1; ++i) {
                rhs_A[j*ldu1+i] = U1[j*ldu1+i];
            }
        }

        dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
        dLUstructInit(m_A, &LUstruct_A);
        PStatInit(&stat_A);
        pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldu1, r1, grid_A,
            &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

        Destroy_CompRowLoc_Matrix_dist(&A);
        dScalePermstructFree(&ScalePermstruct_A);
        dDestroy_LU(m_A, grid_A, &LUstruct_A);
        dLUstructFree(&LUstruct_A);
        dSolveFinalize(&options, &SOLVEstruct_A);

        for (j = 0; j < r1; ++j) {
            for (i = 0; i < ldu1; ++i) {
                localZ[j*ldu1+i] = rhs_A[j*ldu1+i];
                iterZ[j*ldu1+i] = rhs_A[j*ldu1+i];
            }
        }

        // printf("Proc %d in grid_A gets Z after 1st solve.\n", grid_A->iam);
        // for (i = 0; i < ldu1; ++i) {
        //     for (j = 0; j < r1; ++j) {
        //         printf("%f ", localZ[j*ldu1+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
        MPI_Barrier(grid_A->comm);
    }

    if ((iam_A != -1) || (iam_B != -1)) {
        if (iam_A != -1) {
            for (j = 0; j < r2*m_B; ++j) {
                for (i = 0; i < ldu2; ++i) {
                    rhs_B[j*ldu2+i] = U2[j*ldu2+i];
                }
            }

            // printf("Proc %d in grid_A gets RHS for 1st solve.\n", grid_A->iam);
            // for (i = 0; i < ldu2; ++i) {
            //     for (j = 0; j < r2*m_B; ++j) {
            //         printf("%f ", rhs_B[j*ldu2+i]);
            //     }
            //     printf("\n");
            // }
            // fflush(stdout);
        }
        if (iam_B != -1) {
            for (j = 0; j < r2*m_A; ++j) {
                for (i = 0; i < ldu2t; ++i) {
                    rhs_BT[j*ldu2t+i] = U2T[j*ldu2t+i];
                }
            }

            // printf("Proc %d in grid_B gets RHS for 1st solve.\n", grid_B->iam);
            // for (i = 0; i < ldu2t; ++i) {
            //     for (j = 0; j < r2*m_A; ++j) {
            //         printf("%f ", rhs_BT[j*ldu2t+i]);
            //     }
            //     printf("\n");
            // }
            // fflush(stdout);
        }
        adi_ls2(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg,
            grid_A, grid_B, rhs_B, ldu2, rhs_BT, ldu2t, r2, q[0], la, ua, lb, ub, iterW_comb, grid_proc, 0, 1);

        dredistribute_X_twogrids(iterW_comb, iterW, iterW_T, grid_A, grid_B, m_A, m_B, r2, grid_proc, 0, 1);

        if (iam_B != -1) {
            for (k = 0; k < r2; ++k) {
                for (j = 0; j < ldu2t; ++j) {
                    for (i = 0; i < m_A; ++i) {
                        localW_T[k*ldu2t*m_A+j*m_A+i] = iterW_T[k*ldu2t*m_A+i*ldu2t+j];
                    }
                }
            }

            if (iam_B == 0) {
                for (j = 0; j < r2; ++j) {
                    localD[j] = q[0]-p[0];
                }
            }

            // printf("Proc %d in grid_B gets WT after 1st solve.\n", grid_B->iam);
            // for (i = 0; i < ldu2t; ++i) {
            //     for (j = 0; j < r2*m_A; ++j) {
            //         printf("%f ", iterW_T[j*ldu2t+i]);
            //     }
            //     printf("\n");
            // }
            // fflush(stdout);

            MPI_Barrier(grid_B->comm);
        }

        if (iam_A != -1) {
            // printf("Proc %d in grid_A gets W after 1st solve.\n", grid_A->iam);
            // for (i = 0; i < ldu2; ++i) {
            //     for (j = 0; j < r2*m_B; ++j) {
            //         printf("%f ", iterW[j*ldu2+i]);
            //     }
            //     printf("\n");
            // }
            // fflush(stdout);

            MPI_Barrier(grid_A->comm);
        }
    }

    if (iam_C != -1) {
        dcreate_SuperMatrix(&C, grid_C, m_C, m_C, nnz_C, nzval_C_neg, rowind_C_neg, colptr_C_neg, p[0]);

        for (j = 0; j < r2; ++j) {
            for (i = 0; i < ldv2; ++i) {
                rhs_C[j*ldv2+i] = V2[j*ldv2+i];
            }
        }

        dScalePermstructInit(m_C, m_C, &ScalePermstruct_C);
        dLUstructInit(m_C, &LUstruct_C);
        PStatInit(&stat_C);
        pdgssvx(&options, &C, &ScalePermstruct_C, rhs_C, ldv2, r2, grid_C,
            &LUstruct_C, &SOLVEstruct_C, berr_C, &stat_C, &info);

        Destroy_CompRowLoc_Matrix_dist(&C);
        dScalePermstructFree(&ScalePermstruct_C);
        dDestroy_LU(m_C, grid_C, &LUstruct_C);
        dLUstructFree(&LUstruct_C);
        dSolveFinalize(&options, &SOLVEstruct_C);

        for (j = 0; j < r2; ++j) {
            for (i = 0; i < ldv2; ++i) {
                localY[j*ldv2+i] = rhs_C[j*ldv2+i];
                iterY[j*ldv2+i] = rhs_C[j*ldv2+i];
            }
        }

        // printf("Proc %d in grid_C gets Y after 1st solve.\n", grid_C->iam);
        // for (i = 0; i < ldv2; ++i) {
        //     for (j = 0; j < r2; ++j) {
        //         printf("%f ", localY[j*ldv2+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
        MPI_Barrier(grid_C->comm);
    }

    rr2 = r2;
    for (k = 1; k < l; ++k) {
        if (iam_A != -1) {
            dcreate_SuperMatrix(&A, grid_A, m_A, m_A, nnz_A, nzval_A, rowind_A, colptr_A, q[k]);

            for (j = 0; j < r1; ++j) {
                for (i = 0; i < ldu1; ++i) {
                    rhs_A[j*ldu1+i] = (q[k]-p[k-1])*iterZ[j*ldu1+i];
                }
            }

            dScalePermstructInit(m_A, m_A, &ScalePermstruct_A);
            dLUstructInit(m_A, &LUstruct_A);
            PStatInit(&stat_A);
            pdgssvx(&options, &A, &ScalePermstruct_A, rhs_A, ldu1, r1, grid_A,
                &LUstruct_A, &SOLVEstruct_A, berr_A, &stat_A, &info);

            Destroy_CompRowLoc_Matrix_dist(&A);
            dScalePermstructFree(&ScalePermstruct_A);
            dDestroy_LU(m_A, grid_A, &LUstruct_A);
            dLUstructFree(&LUstruct_A);
            dSolveFinalize(&options, &SOLVEstruct_A);

            for (j = 0; j < r1; ++j) {
                for (i = 0; i < ldu1; ++i) {
                    iterZ[j*ldu1+i] += rhs_A[j*ldu1+i];
                    localZ[k*r1*ldu1+j*ldu1+i] = iterZ[j*ldu1+i];
                }
            }
                  
            MPI_Barrier(grid_A->comm);
        }
        
        if ((iam_A != -1) || (iam_B != -1)) {
            if (iam_A != -1) {
                for (j = 0; j < r2*m_B; ++j) {
                    for (i = 0; i < ldu2; ++i) {
                        rhs_B[j*ldu2+i] = (q[k]-p[k-1])*iterW[j*ldu2+i];
                    }
                }
            }
            if (iam_B != -1) {
                for (j = 0; j < r2*m_A; ++j) {
                    for (i = 0; i < ldu2t; ++i) {
                        rhs_BT[j*ldu2t+i] = (q[k]-p[k-1])*iterW_T[j*ldu2t+i];
                    }
                }
            }
            adi_ls2(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B_neg, rowind_B_neg, colptr_B_neg,
                grid_A, grid_B, rhs_B, ldu2, rhs_BT, ldu2t, r2, q[k], la, ua, lb, ub, iterW_comb_tmp, grid_proc, 0, 1);

            if (iam_A == 0) {
                for (j = 0; j < r2; ++j) {
                    for (i = 0; i < m_A*m_B; ++i) {
                        iterW_comb[j*m_A*m_B+i] += iterW_comb_tmp[j*m_A*m_B+i];
                    }
                }
            }
            dredistribute_X_twogrids(iterW_comb, iterW, iterW_T, grid_A, grid_B, m_A, m_B, r2, grid_proc, 0, 1);

            if (iam_B != -1) {
                for (m = 0; m < r2; ++m) {
                    for (j = 0; j < ldu2t; ++j) {
                        for (i = 0; i < m_A; ++i) {
                            localW_T[(rr2+m)*ldu2t*m_A+j*m_A+i] = iterW_T[m*ldu2t*m_A+i*ldu2t+j];
                        }
                    }
                }

                if (iam_B == 0) {
                    for (j = 0; j < r2; ++j) {
                        localD[rr2+j] = q[k]-p[k];
                    }
                }

                // printf("Proc %d in B finishes solving for iteration %d.\n", iam_B, k);
                // fflush(stdout);

                MPI_Barrier(grid_B->comm);
            }

            if (iam_A != -1) {
                // printf("Proc %d in A finishes solving for iteration %d.\n", iam_A, k);
                // fflush(stdout);

                MPI_Barrier(grid_A->comm);
            }
        }

        if (iam_C != -1) {
            dcreate_SuperMatrix(&C, grid_C, m_C, m_C, nnz_C, nzval_C_neg, rowind_C_neg, colptr_C_neg, p[k]);

            for (j = 0; j < r2; ++j) {
                for (i = 0; i < ldv2; ++i) {
                    rhs_C[j*ldv2+i] = (p[k]-q[k-1])*iterY[j*ldv2+i];
                }
            }

            dScalePermstructInit(m_C, m_C, &ScalePermstruct_C);
            dLUstructInit(m_C, &LUstruct_C);
            PStatInit(&stat_C);
            pdgssvx(&options, &C, &ScalePermstruct_C, rhs_C, ldv2, r2, grid_C,
                &LUstruct_C, &SOLVEstruct_C, berr_C, &stat_C, &info);

            Destroy_CompRowLoc_Matrix_dist(&C);
            dScalePermstructFree(&ScalePermstruct_C);
            dDestroy_LU(m_C, grid_C, &LUstruct_C);
            dLUstructFree(&LUstruct_C);
            dSolveFinalize(&options, &SOLVEstruct_C);

            for (j = 0; j < r2; ++j) {
                for (i = 0; i < ldv2; ++i) {
                    iterY[j*ldv2+i] += rhs_C[j*ldv2+i];
                    localY[rr2*ldv2+j*ldv2+i] = iterY[j*ldv2+i];
                }
            }

            // printf("Proc %d in C finishes solving for iteration %d.\n", iam_C, k);
            // fflush(stdout);

            MPI_Barrier(grid_C->comm);
        }

        rr2 += r2;
        double *compressU, *compressS, *compressV;
        drecompression_dist_twogrids(localW_T, m_A*ldu2t, m_A*m_B, &compressU, localD, &compressS, grid_B,
            localY, ldv2, m_C, &compressV, grid_C, &rr2, tol, grid_proc, 1, 2);
        
        if (iam_B == 0) {
            MPI_Send(&rr2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else if (iam_A == 0) {
            MPI_Recv(&rr2, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (iam_A != -1) {
            MPI_Bcast(&rr2, 1, MPI_INT, 0, grid_A->comm);

            // printf("Proc %d in A finishes recompressing for iteration %d.\n", iam_A, k);
            // fflush(stdout);

            MPI_Barrier(grid_A->comm);
        }

        if (iam_B != -1) {
            for (j = 0; j < rr2; ++j) {
                for (i = 0; i < m_A*ldu2t; ++i) {
                    localW_T[j*m_A*ldu2t+i] = compressU[j*ldu2t*m_A+i];
                }
            }
            SUPERLU_FREE(compressU);

            if (iam_B == 0) {
                for (j = 0; j < rr2; ++j) {
                    localD[j] = compressS[j];
                }
                SUPERLU_FREE(compressS);
            }

            // printf("Proc %d in B finishes recompressing for iteration %d.\n", iam_B, k);
            // fflush(stdout);

            MPI_Barrier(grid_B->comm);
        }
        if (iam_C != -1) {
            for (j = 0; j < rr2; ++j) {
                for (i = 0; i < ldv2; ++i) {
                    localY[j*ldv2+i] = compressV[j*ldv2+i];
                }
            }
            SUPERLU_FREE(compressV);

            // printf("Proc %d in C finishes recompressing for iteration %d.\n", iam_C, k);
            // fflush(stdout);

            MPI_Barrier(grid_C->comm);
        }
    }

    *rank2 = rr2;

    // printf("Proc %d in grid_A gets T1 before truncation.\n", grid_A->iam);
    // for (i = 0; i < ldu; ++i) {
    //     for (j = 0; j < r*l; ++j) {
    //         printf("%f ", localZ[j*ldu+i]);
    //     }
    //     printf("\n");
    // }
    // fflush(stdout);
    // MPI_Barrier(grid_A->comm);

    // printf("Proc %d finishes all adi iterations, knowing rank2 %d.\n", global_rank, rr2);
    // fflush(stdout);

    if (iam_A != -1) {
        // dCPQR_dist_getQ(localZ, ldu1, T1, r1*l, rank1, grid_A, tol);
        ovsamp = r1*l >= 20 ? 5 : 2;
        dCPQR_dist_rand_getQ(localZ, ldu1, m_A, T1, r1*l, rank1, grid_A, tol, ovsamp);

        // printf("Proc %d in grid_A gets T1.\n", grid_A->iam);
        // for (i = 0; i < ldu1; ++i) {
        //     for (j = 0; j < *rank1; ++j) {
        //         printf("%f ", (*T1)[j*ldu1+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);

        if (iam_A == 0) {
            if ( !(global_T1 = doubleMalloc_dist(m_A*(*rank1))) )
                ABORT("Malloc fails for global_T1[].");
        }
        dgather_X(*T1, ldu1, global_T1, m_A, *rank1, grid_A);

        MPI_Barrier(grid_A->comm);
    }
    
    if (iam_A == 0) {
        MPI_Send(rank1, 1, MPI_INT, grid_proc[0], 0, MPI_COMM_WORLD);
    }
    else if (iam_B == 0) {
        MPI_Recv(rank1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (iam_B != -1) {
        MPI_Bcast(rank1, 1, MPI_INT, 0, grid_B->comm);

        if ( !(global_T1_onB = doubleMalloc_dist(m_A*(*rank1))) )
            ABORT("Malloc fails for global_T1_onB[].");
    }
    transfer_X_dgrids(global_T1, m_A, *rank1, global_T1_onB, grid_proc, 0, 1);
    if (iam_B != -1) {
        MPI_Bcast(global_T1_onB, m_A*(*rank1), MPI_DOUBLE, 0, grid_B->comm);

        MPI_Bcast(localD, rr2, MPI_DOUBLE, 0, grid_B->comm);
        for (j = 0; j < rr2; ++j) {
            for (i = 0; i < m_A*ldu2t; ++i) {
                localW_T[j*m_A*ldu2t+i] *= localD[j];
            }
        }

        tmpr2 = ldu2t*rr2;
        if ( !(*T2 = doubleMalloc_dist((*rank1)*tmpr2)) )
            ABORT("Malloc fails for *T2[].");
        dgemm_("T", "N", rank1, &(tmpr2), &(m_A), &one, global_T1_onB, &(m_A), 
            localW_T, &(m_A), &zero, *T2, rank1);

        // printf("Proc %d in grid_B gets T2.\n", grid_B->iam);
        // for (i = 0; i < ldu2t*(*rank1); ++i) {
        //     for (j = 0; j < rr2; ++j) {
        //         printf("%f ", (*T2)[j*ldu2t*(*rank1)+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
    }

    if (iam_A == 0) {
        MPI_Send(rank1, 1, MPI_INT, grid_proc[0]+grid_proc[1], 0, MPI_COMM_WORLD);
    }
    else if (iam_C == 0) {
        MPI_Recv(rank1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (iam_C != -1) {
        if ( !(*T3 = doubleMalloc_dist(ldv2*rr2)) )
            ABORT("Malloc fails for *T3[].");
        for (j = 0; j < ldv2; ++j) {
            for (i = 0; i < rr2; ++i) {
                (*T3)[j*rr2+i] = localY[i*ldv2+j];
            }
        }

        // printf("Proc %d in grid_C gets T3.\n", grid_C->iam);
        // for (i = 0; i < rr2; ++i) {
        //     for (j = 0; j < ldv2; ++j) {
        //         printf("%f ", (*T3)[j*rr2+i]);
        //     }
        //     printf("\n");
        // }
        // fflush(stdout);
    }

    // printf("Proc %d in grid_A gets T1 after truncation.\n", grid_A->iam);
    // for (i = 0; i < ldu; ++i) {
    //     for (j = 0; j < *rank; ++j) {
    //         printf("%f ", (*Z)[j*ldu+i]);
    //     }
    //     printf("\n");
    // }
    // fflush(stdout);
    
    if (iam_A != -1) {
        PStatFree(&stat_A);
        SUPERLU_FREE(berr_A);
        SUPERLU_FREE(rhs_A);
        SUPERLU_FREE(localZ);
        SUPERLU_FREE(iterZ);
        SUPERLU_FREE(rhs_B);
        SUPERLU_FREE(iterW);
        SUPERLU_FREE(iterW_comb);

        if (iam_A == 0) {
            SUPERLU_FREE(iterW_comb_tmp);
            SUPERLU_FREE(global_T1);
        }
    }
    if (iam_B != -1) {
        SUPERLU_FREE(iterW_T);
        SUPERLU_FREE(rhs_BT);
        SUPERLU_FREE(localW_T);
        SUPERLU_FREE(global_T1_onB);
        SUPERLU_FREE(localD);
    }
    if (iam_C != -1) {
        PStatFree(&stat_C);
        SUPERLU_FREE(berr_C);
        SUPERLU_FREE(rhs_C);
        SUPERLU_FREE(localY);
        SUPERLU_FREE(iterY);
    }
}