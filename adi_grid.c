/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief SuperLU grid utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * February 8, 2019  version 6.1.1
 * October 5, 2021
 * </pre>
 */

#include "superlu_ddefs.h"
#include "adi_grid.h"

/*! \brief All processes in the MPI communicator must call this routine.
 * 
 *  On output, if a process is not in the SuperLU group, the following 
 *  values are assigned to it:
 *      grid->comm = MPI_COMM_NULL
 *      grid->iam = -1
 */
void adi_gridinit_matrix(MPI_Comm Bcomm,
        int nprow_A, int npcol_A, gridinfo_t *grid_A,
        int nprow_B, int npcol_B, gridinfo_t *grid_B)
{
    int Np_A = nprow_A * npcol_A;
    int Np_B = nprow_B * npcol_B;
    int *usermap_A, *usermap_B;
    int i, j, info;

    /* Make a list of the processes in the new communicator. */
    usermap_A = SUPERLU_MALLOC(Np_A*sizeof(int));
    for (j = 0; j < npcol_A; ++j) {
        for (i = 0; i < nprow_A; ++i) {
            usermap_A[j*nprow_A+i] = i*npcol_A+j;
        }
    }

    usermap_B = SUPERLU_MALLOC(Np_B*sizeof(int));
    for (j = 0; j < npcol_B; ++j) {
        for (i = 0; i < nprow_B; ++i) {
            usermap_B[j*nprow_B+i] = Np_A + i*npcol_B+j;
        }
    }
    
    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
    ABORT("C main program must explicitly call MPI_Init()");

    MPI_Comm_size( Bcomm, &info );
    if ( info < Np_A+Np_B ) {
    printf("Number of processes %d is smaller than combined grid size %d", info, Np_A+Np_B);
    exit(-1);
    }

    adi_gridmap_matrix(Bcomm, nprow_A, npcol_A, usermap_A, nprow_A, grid_A,
        nprow_B, npcol_B, usermap_B, nprow_B, grid_B);
    
    SUPERLU_FREE(usermap_A);
    SUPERLU_FREE(usermap_B);
}


/*! \brief All processes in the MPI communicator must call this routine.
 *
 *  On output, if a process is not in the SuperLU group, the following 
 *  values are assigned to it:
 *      grid->comm = MPI_COMM_NULL
 *      grid->iam = -1
 */
void adi_gridmap_matrix(MPI_Comm Bcomm,
        int nprow_A, int npcol_A, int usermap_A[], int ldumap_A, gridinfo_t *grid_A,
        int nprow_B, int npcol_B, int usermap_B[], int ldumap_B, gridinfo_t *grid_B)
{
    MPI_Group mpi_base_group, superlu_grp_A, superlu_grp_B;
    int Np_A = nprow_A * npcol_A, mycol_A, myrow_A;
    int Np_B = nprow_B * npcol_B, mycol_B, myrow_B;
    int *pranks_A, *pranks_B;
    int i, j, info;
    
    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
    ABORT("C main program must explicitly call MPI_Init()");

    grid_A->nprow = nprow_A;
    grid_A->npcol = npcol_A;
    grid_B->nprow = nprow_B;
    grid_B->npcol = npcol_B;

    /* Make a list of the processes in the new communicator. */
    pranks_A = (int *) SUPERLU_MALLOC(Np_A*sizeof(int));
    for (j = 0; j < npcol_A; ++j) {
        for (i = 0; i < nprow_A; ++i) {
            pranks_A[i*npcol_A+j] = usermap_A[j*ldumap_A+i];
        }
    }

    pranks_B = (int *) SUPERLU_MALLOC(Np_B*sizeof(int));
    for (j = 0; j < npcol_B; ++j) {
        for (i = 0; i < nprow_B; ++i) {
            pranks_B[i*npcol_B+j] = usermap_B[j*ldumap_B+i];
        }
    }
    
    /*
     * Form MPI communicator for all.
     */
    /* Get the group underlying Bcomm. */
    MPI_Comm_group( Bcomm, &mpi_base_group );
    /* Create the new group. */
    MPI_Group_incl( mpi_base_group, Np_A, pranks_A, &superlu_grp_A );
    MPI_Group_incl( mpi_base_group, Np_B, pranks_B, &superlu_grp_B );
    /* Create the new communicator. */
    /* NOTE: The call is to be executed by all processes in Bcomm,
       even if they do not belong in the new group -- superlu_grp.
       The function returns MPI_COMM_NULL to processes that are not in superlu_grp. */
    MPI_Comm_create( Bcomm, superlu_grp_A, &grid_A->comm );
    MPI_Comm_create( Bcomm, superlu_grp_B, &grid_B->comm );

    /* Bail out if I am not in the group "superlu_grp". */
    if (( grid_A->comm == MPI_COMM_NULL ) && ( grid_B->comm == MPI_COMM_NULL )) {
        // grid->comm = Bcomm;  do not need to reassign to a valid communicator
        grid_A->iam = -1;
        grid_B->iam = -1;
        //SUPERLU_FREE(pranks);
        //return;
    }
    else if (( grid_A->comm == MPI_COMM_NULL ) && ( grid_B->comm != MPI_COMM_NULL )){
        grid_A->iam = -1;

        MPI_Comm_rank( grid_B->comm, &(grid_B->iam) );
        myrow_B = grid_B->iam / npcol_B;
        mycol_B = grid_B->iam % npcol_B;

        /*
         * Form MPI communicator for myrow, scope = COMM_ROW.
         */
        MPI_Comm_split(grid_B->comm, myrow_B, mycol_B, &(grid_B->rscp.comm));

        /*
         * Form MPI communicator for mycol, scope = COMM_COLUMN.
         */
        MPI_Comm_split(grid_B->comm, mycol_B, myrow_B, &(grid_B->cscp.comm));

        grid_B->rscp.Np = npcol_B;
        grid_B->rscp.Iam = mycol_B;
        grid_B->cscp.Np = nprow_B;
        grid_B->cscp.Iam = myrow_B;
    }
    else if (( grid_A->comm != MPI_COMM_NULL ) && ( grid_B->comm == MPI_COMM_NULL )) {
        grid_B->iam = -1;

        MPI_Comm_rank( grid_A->comm, &(grid_A->iam) );
        myrow_A = grid_A->iam / npcol_A;
        mycol_A = grid_A->iam % npcol_A;

        /*
         * Form MPI communicator for myrow, scope = COMM_ROW.
         */
        MPI_Comm_split(grid_A->comm, myrow_A, mycol_A, &(grid_A->rscp.comm));

        /*
         * Form MPI communicator for mycol, scope = COMM_COLUMN.
         */
        MPI_Comm_split(grid_A->comm, mycol_A, myrow_A, &(grid_A->cscp.comm));

        grid_A->rscp.Np = npcol_A;
        grid_A->rscp.Iam = mycol_A;
        grid_A->cscp.Np = nprow_A;
        grid_A->cscp.Iam = myrow_A;
    }
    else {
        ABORT("There shouldn't be a process in both grids!\n");
    }

    SUPERLU_FREE(pranks_A);
    SUPERLU_FREE(pranks_B);
    MPI_Group_free(&superlu_grp_A);
    MPI_Group_free(&superlu_grp_B);
    MPI_Group_free(&mpi_base_group);
} /* superlu_gridmap */

void adi_gridexit_matrix(gridinfo_t *grid_A, gridinfo_t *grid_B)
{
    if ( grid_A->comm != MPI_COMM_NULL ) {
        /* Marks the communicator objects for deallocation. */
        MPI_Comm_free( &grid_A->rscp.comm );
        MPI_Comm_free( &grid_A->cscp.comm );
        MPI_Comm_free( &grid_A->comm );
    }
    if ( grid_B->comm != MPI_COMM_NULL ) {
        /* Marks the communicator objects for deallocation. */
        MPI_Comm_free( &grid_B->rscp.comm );
        MPI_Comm_free( &grid_B->cscp.comm );
        MPI_Comm_free( &grid_B->comm );
    }
}

void adi_gridinit_tensor(MPI_Comm Bcomm, int d,
        int *nprow, int *npcol, int *Np, gridinfo_t **grid)
{
    int **usermap;
    int i, j, k, info, total_Np = 0;

    usermap = SUPERLU_MALLOC(d*sizeof(int*));

    for (k = 0; k < d; ++k) {
        usermap[k] = SUPERLU_MALLOC(Np[k]*sizeof(int));
        for (j = 0; j < npcol[k]; ++j) {
            for (i = 0; i < nprow[k]; ++i) {
                (usermap[k])[j*nprow[k]+i] = total_Np + i*npcol[k]+j;
            }
        }
        total_Np += Np[k];
    }
    
    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
        ABORT("C main program must explicitly call MPI_Init()");

    MPI_Comm_size( Bcomm, &info );
    if ( info < total_Np ) {
        printf("Number of processes %d is smaller than combined grid size %d", info, total_Np);
        exit(-1);
    }

    adi_gridmap_tensor(Bcomm, d, nprow, npcol, Np, usermap, nprow, grid);
    
    for (k = 0; k < d; ++k) {
        SUPERLU_FREE(usermap[k]);
    }
    SUPERLU_FREE(usermap);
}

void adi_gridmap_tensor(MPI_Comm Bcomm, int d,
        int *nprow, int *npcol, int *Np, int **usermap, int *ldumap, gridinfo_t **grid)
{
    MPI_Group mpi_base_group;
    MPI_Group *superlu_grp;
    int mycol, myrow;
    int **pranks;
    int i, j, k, info;
    int in_group = -1;
    
    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
        ABORT("C main program must explicitly call MPI_Init()");

    pranks = (int **) SUPERLU_MALLOC(d*sizeof(int*));

    for (k = 0; k < d; ++k) {
        grid[k]->nprow = nprow[k];
        grid[k]->npcol = npcol[k];

        pranks[k] = (int *) SUPERLU_MALLOC(Np[k]*sizeof(int));
        for (j = 0; j < npcol[k]; ++j) {
            for (i = 0; i < nprow[k]; ++i) {
                (pranks[k])[i*npcol[k]+j] = (usermap[k])[j*ldumap[k]+i];
            }
        }
    }
    
    /*
     * Form MPI communicator for all.
     */
    /* Get the group underlying Bcomm. */
    MPI_Comm_group( Bcomm, &mpi_base_group );
    
    superlu_grp = (MPI_Group *) SUPERLU_MALLOC(d*sizeof(MPI_Group));
    for (k = 0; k < d; ++k) {
        /* Create the new group. */
        MPI_Group_incl( mpi_base_group, Np[k], pranks[k], &(superlu_grp[k]) );
        /* Create the new communicator. */
        /* NOTE: The call is to be executed by all processes in Bcomm,
           even if they do not belong in the new group -- superlu_grp.
           The function returns MPI_COMM_NULL to processes that are not in superlu_grp. */
        MPI_Comm_create( Bcomm, superlu_grp[k], &grid[k]->comm );
    }

    for (k = 0; k < d; ++k) {
        if (grid[k]->comm == MPI_COMM_NULL) {
            grid[k]->iam = -1;
        }
        else {
            if (in_group != -1) {
                ABORT("There shouldn't be a process in both grids!\n");
            }
            else {
                in_group = k;

                MPI_Comm_rank( grid[k]->comm, &(grid[k]->iam) );
                myrow = grid[k]->iam / npcol[k];
                mycol = grid[k]->iam % npcol[k];

                /*
                 * Form MPI communicator for myrow, scope = COMM_ROW.
                 */
                MPI_Comm_split(grid[k]->comm, myrow, mycol, &(grid[k]->rscp.comm));

                /*
                 * Form MPI communicator for mycol, scope = COMM_COLUMN.
                 */
                MPI_Comm_split(grid[k]->comm, mycol, myrow, &(grid[k]->cscp.comm));

                grid[k]->rscp.Np = npcol[k];
                grid[k]->rscp.Iam = mycol;
                grid[k]->cscp.Np = nprow[k];
                grid[k]->cscp.Iam = myrow;
            }
        }
    }

    for (k = 0; k < d; ++k) {
        SUPERLU_FREE(pranks[k]);
        MPI_Group_free(&(superlu_grp[k]));
    }
    SUPERLU_FREE(pranks);
    SUPERLU_FREE(superlu_grp);
    MPI_Group_free(&mpi_base_group);
}

void adi_gridexit_tensor(gridinfo_t **grid, int d)
{
    int_t k;
    for (k = 0; k < d; ++k) {
        if ( grid[k]->comm != MPI_COMM_NULL ) {
        /* Marks the communicator objects for deallocation. */
            MPI_Comm_free( &grid[k]->rscp.comm );
            MPI_Comm_free( &grid[k]->cscp.comm );
            MPI_Comm_free( &grid[k]->comm );
        }
    }
}