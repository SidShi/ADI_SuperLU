#include <math.h>
#include "superlu_ddefs.h"
#include "read_equation.h"

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

void dread_matrix(FILE *fp, char * postfix, gridinfo_t *grid, int_t *m, int_t *n, 
    int_t *nnz, double **nzval, int_t **rowind, int_t **colptr)
{
    int_t i;
    int_t chunk= 2000000000;
    int count;
    int_t Nchunk;
    int_t remainder;
    int iam = grid->iam;

    if ( !iam ) {
    double t = SuperLU_timer_(); 

    if(!strcmp(postfix,"rua")){
        /* Read the matrix stored on disk in Harwell-Boeing format. */
        dreadhb_dist(iam, fp, m, n, nnz, nzval, rowind, colptr);
    }else if(!strcmp(postfix,"mtx")){
        /* Read the matrix stored on disk in Matrix Market format. */
        dreadMM_dist(fp, m, n, nnz, nzval, rowind, colptr);
    }else if(!strcmp(postfix,"rb")){
        /* Read the matrix stored on disk in Rutherford-Boeing format. */
        dreadrb_dist(iam, fp, m, n, nnz, nzval, rowind, colptr);      
    }else if(!strcmp(postfix,"dat")){
        /* Read the matrix stored on disk in triplet format. */
        dreadtriple_dist(fp, m, n, nnz, nzval, rowind, colptr);
    }else if(!strcmp(postfix,"datnh")){
        /* Read the matrix stored on disk in triplet format (without header). */
        dreadtriple_noheader(fp, m, n, nnz, nzval, rowind, colptr);       
    }else if(!strcmp(postfix,"bin")){
        /* Read the matrix stored on disk in binary format. */
        dread_binary(fp, m, n, nnz, nzval, rowind, colptr);       
    }else {
        ABORT("File format not known");
    }

    printf("Time to read and distribute matrix %.2f\n", 
            SuperLU_timer_() - t);  fflush(stdout);
            
    /* Broadcast matrix A to the other PEs. */
    MPI_Bcast( m,     1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( n,     1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( nnz,   1,   mpi_int_t,  0, grid->comm );

    
    Nchunk = CEILING(*nnz,chunk);
    remainder =  (*nnz)%chunk;
    MPI_Bcast( &Nchunk,   1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( &remainder,   1,   mpi_int_t,  0, grid->comm );

    for (i = 0; i < Nchunk; ++i) {
       int_t idx=i*chunk;
       if(i==Nchunk-1){
            count=remainder;
       }else{
            count=chunk;
       }  
        MPI_Bcast( nzval[idx],  count, MPI_DOUBLE, 0, grid->comm );
        MPI_Bcast( rowind[idx], count, mpi_int_t,  0, grid->comm );       
    }




    MPI_Bcast( *colptr, (*n)+1, mpi_int_t,  0, grid->comm );
    } else if (iam != -1) {
    /* Receive matrix A from PE 0. */
    MPI_Bcast( m,   1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( n,   1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( nnz, 1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( &Nchunk,   1,   mpi_int_t,  0, grid->comm );
    MPI_Bcast( &remainder,   1,   mpi_int_t,  0, grid->comm );


    /* Allocate storage for compressed column representation. */
    dallocateA_dist(*n, *nnz, nzval, rowind, colptr);

    for (i = 0; i < Nchunk; ++i) {
       int_t idx=i*chunk;
       if(i==Nchunk-1){
            count=remainder;
       }else{
            count=chunk;
       }  
        MPI_Bcast( nzval[idx],  count, MPI_DOUBLE, 0, grid->comm );
        MPI_Bcast( rowind[idx], count, mpi_int_t,  0, grid->comm );       
    }
    MPI_Bcast( *colptr,  (*n)+1, mpi_int_t,  0, grid->comm );
    }
}

void dread_shift(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B)
{
    int_t i;
    // int_t *l1;
    int l1, l2;
    int nproc_A = grid_A->nprow * grid_A->npcol;
    double *rp, *rq;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    // printf("Process with id %d in A and id %d in B gets here.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (grid_A->iam == 0) {
        fscanf(fp, "%d%d\n", &l1, &l2);
        // printf("Process with id %d in A gets the shifts with length %d.\n", grid_A->iam, l1);
        // fflush(stdout);

        *l = l1;
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            fscanf(fp, "%lf%lf\n", &((*p)[i]), &((*q)[i]));
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_A->comm);
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_A->comm);
    }

    // if (grid_A->iam == -1) {
    //     printf("Process with id %d in B gets here.\n", grid_B->iam);
    // }
    // if (grid_A->iam == 1) {
    //     printf("Process with id %d in A gets the shifts with length %d.\n", grid_A->iam, *l);
    //     for (int ll = 0; ll < *l; ll++) {
    //         printf("Shifts are %lf and %lf.\n", p[ll], q[ll]);
    //     }
    // }
    // printf("Process with id %d in A and id %d in B gets here 1.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (!global_rank) {
        MPI_Send(&l1, 1, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == nproc_A) {
        MPI_Recv(&l1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // printf("Process with id %d in A and id %d in B gets here 2.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (grid_B->iam == 0) {
        rp = (double *) doubleMalloc_dist(l1);
        rq = (double *) doubleMalloc_dist(l1);
    }
    transfer_X(*p, l1, 1, rp, grid_A, 1);
    transfer_X(*q, l1, 1, rq, grid_A, 1);

    // printf("Process with id %d in A and id %d in B gets here 3.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    // if (grid_B->iam == 0) {
    //     printf("Process with id %d in B gets the shifts with length %d.\n", grid_B->iam, l1);
    //     for (int ll = 0; ll < l1; ll++) {
    //         printf("Shifts are %lf and %lf.\n", rp[ll], rq[ll]);
    //     }
    // }


    if (grid_B->iam == 0) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            (*p)[i] = rp[i];
            (*q)[i] = rq[i];
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_B->comm);
    }

    // printf("Process with id %d in A and id %d in B gets first shift %f and %f.\n", 
    //     grid_A->iam, grid_B->iam, (*p)[0], (*q)[0]);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    *l = l1;
}

void dread_shift_twogrids(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int grA, int grB, int *grid_proc)
{
    int_t i;
    // int_t *l1;
    int l1, l2;
    double *rp, *rq;
    int global_rank, rootA = 0, rootB = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (grid_A->iam == 0) {
        fscanf(fp, "%d%d\n", &l1, &l2);
        // printf("Process with id %d in A gets the shifts with length %d.\n", grid_A->iam, l1);
        // fflush(stdout);

        *l = l1;
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            fscanf(fp, "%lf%lf\n", &((*p)[i]), &((*q)[i]));
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_A->comm);
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_A->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_A->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_A->comm);
    }

    for (i = 0; i < grA; ++i) {
        rootA += grid_proc[i];
    }
    for (i = 0; i < grB; ++i) {
        rootB += grid_proc[i];
    }

    if (global_rank == rootA) {
        MPI_Send(&l1, 1, MPI_INT, rootB, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == rootB) {
        MPI_Recv(&l1, 1, MPI_INT, rootA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid_B->iam == 0) {
        rp = (double *) doubleMalloc_dist(l1);
        rq = (double *) doubleMalloc_dist(l1);
    }
    transfer_X_dgrids(*p, l1, 1, rp, grid_proc, grA, grB);
    transfer_X_dgrids(*q, l1, 1, rq, grid_proc, grA, grB);

    if (grid_B->iam == 0) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            (*p)[i] = rp[i];
            (*q)[i] = rq[i];
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid_B->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid_B->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid_B->comm);
    }

    *l = l1;
}

void dread_shift_multigrids(FILE *fp, double **p, double **q, int_t *l, gridinfo_t **grids, int *grid_proc, int d)
{
    int_t i, j;
    // int_t *l1;
    int l1, l2;
    double *rp, *rq;
    int global_rank, rootR;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (grids[0]->iam == 0) {
        fscanf(fp, "%d%d\n", &l1, &l2);
        // printf("Process with id %d in A gets the shifts with length %d.\n", grid_A->iam, l1);
        // fflush(stdout);

        *l = l1;
        MPI_Bcast(&l1, 1, MPI_INT, 0, grids[0]->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            fscanf(fp, "%lf%lf\n", &((*p)[i]), &((*q)[i]));
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grids[0]->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grids[0]->comm);
    }
    else if (grids[0]->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grids[0]->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grids[0]->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grids[0]->comm);
    }

    for (j = 1; j < d; ++j) {
        rootR = 0;
        for (i = 0; i < j; ++i) {
            rootR += grid_proc[i];
        }

        if (global_rank == 0) {
            MPI_Send(&l1, 1, MPI_INT, rootR, j, MPI_COMM_WORLD);
        }
        else if (global_rank == rootR) {
            MPI_Recv(&l1, 1, MPI_INT, 0, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (grids[j]->iam == 0) {
            rp = (double *) doubleMalloc_dist(l1);
            rq = (double *) doubleMalloc_dist(l1);
        }
        transfer_X_dgrids(*p, l1, 1, rp, grid_proc, 0, j);
        transfer_X_dgrids(*q, l1, 1, rq, grid_proc, 0, j);

        if (grids[j]->iam == 0) {
            MPI_Bcast(&l1, 1, MPI_INT, 0, grids[j]->comm);

            *p = (double *) doubleMalloc_dist(l1);
            *q = (double *) doubleMalloc_dist(l1);
            for (i = 0; i < l1; ++i) {
                (*p)[i] = rp[i];
                (*q)[i] = rq[i];
            }

            MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grids[j]->comm);
            MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grids[j]->comm);

            SUPERLU_FREE(rp);
            SUPERLU_FREE(rq);
        }
        else if (grids[j]->iam != -1) {
            MPI_Bcast(&l1, 1, MPI_INT, 0, grids[j]->comm);

            *p = (double *) doubleMalloc_dist(l1);
            *q = (double *) doubleMalloc_dist(l1);

            MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grids[j]->comm);
            MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grids[j]->comm);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    *l = l1;
}

void dread_shift_onegrid(FILE *fp, double **p, double **q, int_t *l, gridinfo_t *grid)
{
    int_t i;
    int l1, l2;
    double *rp, *rq;

    if (grid->iam == 0) {
        fscanf(fp, "%d%d\n", &l1, &l2);

        *l = l1;
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);
        for (i = 0; i < l1; ++i) {
            fscanf(fp, "%lf%lf\n", &((*p)[i]), &((*q)[i]));
        }

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid->comm);
    }
    else if (grid->iam != -1) {
        MPI_Bcast(&l1, 1, MPI_INT, 0, grid->comm);

        *p = (double *) doubleMalloc_dist(l1);
        *q = (double *) doubleMalloc_dist(l1);

        MPI_Bcast(*p, l1, MPI_DOUBLE, 0, grid->comm);
        MPI_Bcast(*q, l1, MPI_DOUBLE, 0, grid->comm);
    }

    *l = l1;
}

void dread_shift_interval(FILE *fp, double *a, double *b, double *c, double *d, gridinfo_t *grid)
{
    if (grid->iam == 0) {
        fscanf(fp, "%lf%lf%lf%lf\n", a, b, c, d);
        // printf("Process with id %d in A gets the shifts with length %d.\n", grid_A->iam, l1);
    }
}

void dread_shift_interval_twogrids(FILE *fp, double *a, double *b, double *c, double *d, gridinfo_t *grid_A, gridinfo_t *grid_B,
    int grA, int grB, int *grid_proc)
{
    double abcd[4], r_abcd[4];
    if (grid_A->iam == 0) {
        fscanf(fp, "%lf%lf%lf%lf\n", a, b, c, d);
        abcd[0] = *a; abcd[1] = *b; abcd[2] = *c; abcd[3] = *d;
    }
    transfer_X_dgrids(abcd, 4, 1, r_abcd, grid_proc, grA, grB);

    if (grid_B->iam == 0) {
        *a = r_abcd[0]; *b = r_abcd[1]; *c = r_abcd[2]; *d = r_abcd[3];
    }
}

void dread_shift_interval_multigrids(FILE *fp, int d, double *la, double *ua, double *lb, double *ub, gridinfo_t **grids, int *grid_proc)
{
    int_t i;
    double *rla, *rua, *rlb, *rub;

    if (grids[0]->iam == 0) {
        for (i = 0; i < d-2; ++i) {
            fscanf(fp, "%lf%lf%lf%lf\n", &(la[i]), &(ua[i]), &(lb[i]), &(ub[i]));
        }
    }

    for (j = 1; j < d; ++j) {
        if (grids[j]->iam == 0) {
            rla = (double *) doubleMalloc_dist(d-2);
            rua = (double *) doubleMalloc_dist(d-2);
            rlb = (double *) doubleMalloc_dist(d-2);
            rub = (double *) doubleMalloc_dist(d-2);
        }
        transfer_X_dgrids(la, d-2, 1, rla, grid_proc, 0, j);
        transfer_X_dgrids(ua, d-2, 1, rua, grid_proc, 0, j);
        transfer_X_dgrids(lb, d-2, 1, rlb, grid_proc, 0, j);
        transfer_X_dgrids(ub, d-2, 1, rub, grid_proc, 0, j);

        if (grids[j]->iam == 0) {
            for (i = 0; i < d-2; ++i) {
                la[i] = rla[i];
                ua[i] = rua[i];
                lb[i] = rlb[i];
                ub[i] = rub[i];
            }

            SUPERLU_FREE(rla);
            SUPERLU_FREE(rua);
            SUPERLU_FREE(rlb);
            SUPERLU_FREE(rub);
        }

        if (grids[0]->iam != -1) {
            MPI_Barrier(grids[0]->comm);
        }
        if (grids[j]->iam != -1) {
            MPI_Barrier(grids[j]->comm);
        }
    }
}

void dread_size(FILE *fp, int d, int *ms, gridinfo_t **grids)
{
    int_t i;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    if (grids[0]->iam == 0) {
        for (i = 0; i < d; ++i) {
            fscanf(fp, "%d\n", &(ms[i]));
        }
    }
    MPI_Bcast(ms, d, MPI_INT, 0, MPI_COMM_WORLD);
}

void dread_X(FILE *fp, double **X, gridinfo_t *grid)
{
    int_t i, j;
    int m, n;
    int iam = grid->iam;

    if (iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);

        (*X) = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*X)[i]));
        }
    }
}

void dread_RHS(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t *m_A, int_t *m_B, int *ldf, int *ldft, double **F_global)
{
    int_t i, j;
    int m, n;
    int_t m_loc, m_loc_fst, row, fst_row;
    int *dim;
    int iam;
    int nproc_A = grid_A->nprow * grid_A->npcol;
    int nproc_B = grid_B->nprow * grid_B->npcol;
    double *rF_global;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    dim = (int *) intMalloc_dist(2);

    if (grid_A->iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_A->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*F_global)[i]));
        }
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid_A->comm);

        dim[0] = m;
        dim[1] = n;
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_A->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid_A->comm);
    }

    // printf("Process with id %d in A and id %d in B gets here 1.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (!global_rank) {
        MPI_Send(dim, 2, MPI_INT, nproc_A, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == nproc_A) {
        MPI_Recv(dim, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid_B->iam == 0) {
        m = dim[0];
        n = dim[1];
        rF_global = (double *) doubleMalloc_dist(m*n);
    }
    transfer_X(*F_global, m*n, 1, rF_global, grid_A, 1);

    // printf("Process with id %d in A and id %d in B gets here 2.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (grid_B->iam == 0) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_B->comm);

        MPI_Bcast(rF_global, m*n, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_B->comm);

        rF_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(rF_global, m*n, MPI_DOUBLE, 0, grid_B->comm);
    }

    *m_A = m;
    *m_B = n;

    // printf("Process with id %d in A and id %d in B gets here 3.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    // printf("Process with id %d in A and id %d in B gets global F with element count %d.\n", grid_A->iam, grid_B->iam, m*n);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    if (grid_A->iam != -1) {
        m_loc = m / nproc_A; 
        m_loc_fst = m_loc;
        iam = grid_A->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_A) != m) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_A - 1)) /* last proc. gets all*/
          m_loc = m - m_loc * (nproc_A - 1);
        }

        /* Get the local B */
        if ( !((*F) = doubleMalloc_dist(m_loc*n)) )
            ABORT("Malloc fails for F[]");
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m_loc; ++i) {
                row = fst_row + i;
                (*F)[j*m_loc+i] = (*F_global)[j*m+row];
            }
        }
        *ldf = m_loc;

        // printf("Process with id %d in A gets local F with element count %d.\n", grid_A->iam, m_loc*n);
        // fflush(stdout);
    }

    if (grid_B->iam != -1) {
        m_loc = n / nproc_B; 
        m_loc_fst = m_loc;
        iam = grid_B->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_B) != n) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_B - 1)) /* last proc. gets all*/
          m_loc = n - m_loc * (nproc_B - 1);
        }

        /* Get the local B */
        if ( !((*F_transpose) = doubleMalloc_dist(m_loc*m)) )
            ABORT("Malloc fails for F_transpose[]");
        for (j = 0; j < m; ++j) {
            for (i = 0; i < m_loc; ++i) {
                row = fst_row + i;
                (*F_transpose)[j*m_loc+i] = rF_global[row*m+j];
            }
        }
        *ldft = m_loc;

        SUPERLU_FREE(rF_global);

        // printf("Process with id %d in B gets local F with element count %d.\n", grid_B->iam, m_loc*m);
        // fflush(stdout);
    }

    SUPERLU_FREE(dim);

    // printf("Process with id %d in A and id %d in B gets here 4.\n", grid_A->iam, grid_B->iam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);
}

void dread_RHS_multiple(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid_A, gridinfo_t *grid_B, 
    int_t *m_A, int_t *m_B, int_t *r, int *ldf, int *ldft, int *grid_proc, int send_grid, int recv_grid)
{
    int_t i, j, k;
    int m, n, rr;
    int_t m_loc, m_loc_fst, row, fst_row;
    int *dim;
    int iam, grA = 0, grB = 0;
    int nproc_A = grid_A->nprow * grid_A->npcol;
    int nproc_B = grid_B->nprow * grid_B->npcol;
    double *F_global, *rF_global;
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    dim = (int *) intMalloc_dist(3);

    if (grid_A->iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        fscanf(fp, "%d\n", &rr);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&rr, 1, MPI_INT, 0, grid_A->comm);

        F_global = (double *) doubleMalloc_dist(m*n*rr);
        for (i = 0; i < m*n*rr; ++i) {
            fscanf(fp, "%lf\n", &(F_global[i]));
        }
        MPI_Bcast(F_global, m*n*rr, MPI_DOUBLE, 0, grid_A->comm);

        dim[0] = m;
        dim[1] = n;
        dim[2] = rr;
    }
    else if (grid_A->iam != -1) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_A->comm);
        MPI_Bcast(&rr, 1, MPI_INT, 0, grid_A->comm);

        F_global = (double *) doubleMalloc_dist(m*n*rr);
        MPI_Bcast(F_global, m*n*rr, MPI_DOUBLE, 0, grid_A->comm);
    }

    for (j = 0; j < send_grid; ++j) {
        grA += grid_proc[j];
    }
    for (j = 0; j < recv_grid; ++j) {
        grB += grid_proc[j];
    }

    if (global_rank == grA) {
        MPI_Send(dim, 3, MPI_INT, grB, 0, MPI_COMM_WORLD);
    }
    else if (global_rank == grB) {
        MPI_Recv(dim, 3, MPI_INT, grA, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (grid_B->iam == 0) {
        m = dim[0];
        n = dim[1];
        rr = dim[2];
        rF_global = (double *) doubleMalloc_dist(m*n*rr);
    }
    transfer_X_dgrids(F_global, m*n, rr, rF_global, grid_proc, send_grid, recv_grid);

    if (grid_B->iam == 0) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&rr, 1, MPI_INT, 0, grid_B->comm);

        MPI_Bcast(rF_global, m*n*rr, MPI_DOUBLE, 0, grid_B->comm);
    }
    else if (grid_B->iam != -1) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid_B->comm);
        MPI_Bcast(&rr, 1, MPI_INT, 0, grid_B->comm);

        rF_global = (double *) doubleMalloc_dist(m*n*rr);
        MPI_Bcast(rF_global, m*n*rr, MPI_DOUBLE, 0, grid_B->comm);
    }

    *m_A = m;
    *m_B = n;
    *r = rr;

    if (grid_A->iam != -1) {
        m_loc = m / nproc_A; 
        m_loc_fst = m_loc;
        iam = grid_A->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_A) != m) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_A - 1)) /* last proc. gets all*/
          m_loc = m - m_loc * (nproc_A - 1);
        }

        /* Get the local B */
        if ( !((*F) = doubleMalloc_dist(m_loc*n*rr)) )
            ABORT("Malloc fails for F[]");
        for (k = 0; k < rr; ++k) {
            for (j = 0; j < n; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;
                    (*F)[k*m_loc*n+j*m_loc+i] = F_global[k*m*n+j*m+row];
                }
            }
        }
        *ldf = m_loc;
    }

    if (grid_B->iam != -1) {
        m_loc = n / nproc_B; 
        m_loc_fst = m_loc;
        iam = grid_B->iam;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc_B) != n) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc_B - 1)) /* last proc. gets all*/
          m_loc = n - m_loc * (nproc_B - 1);
        }

        /* Get the local B */
        if ( !((*F_transpose) = doubleMalloc_dist(m_loc*m*rr)) )
            ABORT("Malloc fails for F_transpose[]");
        for (k = 0; k < rr; ++k) {
            for (j = 0; j < m; ++j) {
                for (i = 0; i < m_loc; ++i) {
                    row = fst_row + i;
                    (*F_transpose)[k*m_loc*m+j*m_loc+i] = rF_global[k*m*n+row*m+j];
                }
            }
        }
        *ldft = m_loc;

        SUPERLU_FREE(rF_global);
    }

    SUPERLU_FREE(dim);
}

void dread_RHS_onegrid(FILE *fp, double **F, double **F_transpose, gridinfo_t *grid, 
    int_t *m_A, int_t *m_B, int *ldf, int *ldft, double **F_global)
{
    int_t i, j;
    int m, n;
    int_t m_loc, m_loc_fst, row, fst_row;
    int iam = grid->iam;
    int nproc = grid->nprow * grid->npcol;

    if (iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*F_global)[i]));
        }
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }
    else if (iam != -1) {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }

    *m_A = m;
    *m_B = n;

    if (iam != -1) {
        m_loc = m / nproc; 
        m_loc_fst = m_loc;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc) != m) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc - 1)) /* last proc. gets all*/
              m_loc = m - m_loc * (nproc - 1);
        }

        /* Get the local B */
        if ( !((*F) = doubleMalloc_dist(m_loc*n)) )
            ABORT("Malloc fails for F[]");
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m_loc; ++i) {
                row = fst_row + i;
                (*F)[j*m_loc+i] = (*F_global)[j*m+row];
            }
        }
        *ldf = m_loc;

        
        m_loc = n / nproc; 
        m_loc_fst = m_loc;
        fst_row = iam * m_loc_fst;
        /* When nrhs / procs is not an integer */
        if ((m_loc * nproc) != n) {
            /*m_loc = m_loc+1;
              m_loc_fst = m_loc;*/
          if (iam == (nproc - 1)) /* last proc. gets all*/
              m_loc = n - m_loc * (nproc - 1);
        }

        /* Get the local B */
        if ( !((*F_transpose) = doubleMalloc_dist(m_loc*m)) )
            ABORT("Malloc fails for F_transpose[]");
        for (j = 0; j < m; ++j) {
            for (i = 0; i < m_loc; ++i) {
                row = fst_row + i;
                (*F_transpose)[j*m_loc+i] = (*F_global)[row*m+j];
            }
        }
        *ldft = m_loc;
    }
}

void dread_RHS_factor(FILE *fp, double **F, gridinfo_t *grid, int_t *m_global, int_t *r, int *ldf, double **F_global)
{
    int_t i, j;
    int m, n;
    int_t m_loc, m_loc_fst, row, fst_row;
    int nproc = grid->nprow * grid->npcol;
    int iam;

    if (grid->iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*F_global)[i]));
        }
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }
    else {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }

    *m_global = m;
    *r = n;

    m_loc = m / nproc; 
    m_loc_fst = m_loc;
    iam = grid->iam;
    fst_row = iam * m_loc_fst;
    /* When nrhs / procs is not an integer */
    if ((m_loc * nproc) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (nproc - 1)) /* last proc. gets all*/
      m_loc = m - m_loc * (nproc - 1);
    }

    /* Get the local B */
    if ( !((*F) = doubleMalloc_dist(m_loc*n)) )
        ABORT("Malloc fails for F[]");
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m_loc; ++i) {
            row = fst_row + i;
            (*F)[j*m_loc+i] = (*F_global)[j*m+row];
        }
    }
    *ldf = m_loc;
}

void dread_RHS_factor_twodim(FILE *fp, double **F, gridinfo_t *grid, int_t m_A, int *ldf, double **F_global)
{
    int_t i, j;
    int m, n;
    int_t m_B, m_loc_dim, m_loc, m_loc_fst, row, fst_row;
    int nproc = grid->nprow * grid->npcol;
    int iam;

    if (grid->iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*F_global)[i]));
        }
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }
    else {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }

    m_B = m / m_A;

    m_loc_dim = m_B / nproc;
    m_loc = m_loc_dim * m_A;
    m_loc_fst = m_loc;
    iam = grid->iam;
    fst_row = iam * m_loc_fst;
    /* When nrhs / procs is not an integer */
    if ((m_loc_dim * nproc) != m_B) {
      if (iam == (nproc - 1)) /* last proc. gets all*/
        m_loc_dim = m_B - m_loc_dim * (nproc - 1);
        m_loc = m_loc_dim * m_A;
    }

    /* Get the local B */
    if ( !((*F) = doubleMalloc_dist(m_loc*n)) )
        ABORT("Malloc fails for F[]");
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m_loc; ++i) {
            row = fst_row + i;
            (*F)[j*m_loc+i] = (*F_global)[j*m+row];
        }
    }
    *ldf = m_loc;
}

void dread_RHS_factor_multidim(FILE *fp, double **F, gridinfo_t *grid, int_t *ms, int ddeal, int *local, double **F_global)
{
    int_t i, j;
    int m, n;
    int_t m_prior, m_local, m_loc_dim, m_loc, m_loc_fst, row, fst_row;
    int nproc = grid->nprow * grid->npcol;
    int iam;

    m_prior = 1;
    for (j = 0; j < ddeal; ++j) {
        m_prior *= ms[j];
    }

    if (grid->iam == 0) {
        fscanf(fp, "%d\n", &m);
        fscanf(fp, "%d\n", &n);
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        for (i = 0; i < m*n; ++i) {
            fscanf(fp, "%lf\n", &((*F_global)[i]));
        }
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }
    else {
        MPI_Bcast(&m, 1, MPI_INT, 0, grid->comm);
        MPI_Bcast(&n, 1, MPI_INT, 0, grid->comm);

        *F_global = (double *) doubleMalloc_dist(m*n);
        MPI_Bcast(*F_global, m*n, MPI_DOUBLE, 0, grid->comm);
    }

    m_local = ms[ddeal];

    m_loc_dim = m_local / nproc;
    m_loc = m_loc_dim * m_prior;
    m_loc_fst = m_loc;
    iam = grid->iam;
    fst_row = iam * m_loc_fst;
    /* When nrhs / procs is not an integer */
    if ((m_loc_dim * nproc) != m_local) {
      if (iam == (nproc - 1)) /* last proc. gets all*/
        m_loc_dim = m_local - m_loc_dim * (nproc - 1);
        m_loc = m_loc_dim * m_prior;
    }

    /* Get the local B */
    if ( !((*F) = doubleMalloc_dist(m_loc*n)) )
        ABORT("Malloc fails for F[]");
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m_loc; ++i) {
            row = fst_row + i;
            (*F)[j*m_loc+i] = (*F_global)[j*m+row];
        }
    }
    *local = m_loc_dim;
}