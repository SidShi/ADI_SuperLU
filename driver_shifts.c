#include <math.h>
#include "superlu_ddefs.h"
#include "adi.h"
#include "adi_grid.h"
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

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    gridinfo_t grid_A, grid_B;
    int    m_A, m_B;
    int    nprow_A, npcol_A, nprow_B, npcol_B, lookahead, colperm, rowperm, ir, symbfact;
    int    iam_A, iam_B, info, ldf, ldft, ldx;
    char   **cpp, c, *suffix;
    char   postfix[10];
    FILE   *fp_A, *fp_B, *fp_shift, *fp_F, *fp_X, *fopen();
    int ii, omp_mpi_level;
    int ldumap, myrank, p; /* The following variables are used for batch solves */
    int*    usermap;
    double *pp, *qq;
    double *F, *F_transpose, *X, *trueX_global, *F_global;
    int_t ll;
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;
    double i1, i2, i3, i4;

    nprow_A = 1;  /* Default process rows.      */
    npcol_A = 1;  /* Default process columns.   */
    nprow_B = 1;  /* Default process rows.      */
    npcol_B = 1;  /* Default process columns.   */
    lookahead = -1;
    colperm = -1;
    rowperm = -1;
    ir = -1;
    symbfact = -1;

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------*/
    //MPI_Init( &argc, &argv );
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level);
	
    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = SLU_DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES;
	options.DiagInv           = NO;
     */
    set_default_options_dist(&options);
    options.ReplaceTinyPivot = YES;
    options.IterRefine = NOREFINE;
    options.DiagInv           = YES;

    /* Parse command line argv[], may modify default options */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows A     (default %4d)\n", nprow_A);
		  printf("\t-c <int>: process columns A  (default %4d)\n", npcol_A);
          printf("\t-w <int>: process rows B     (default %4d)\n", nprow_B);
          printf("\t-v <int>: process columns B  (default %4d)\n", npcol_B);
		  printf("\t-p <int>: row permutation    (default %4d)\n", options.RowPerm);
		  printf("\t-q <int>: col permutation    (default %4d)\n", options.ColPerm);
		  printf("\t-s <int>: parallel symbolic? (default %4d)\n", options.ParSymbFact);
		  printf("\t-l <int>: lookahead level    (default %4d)\n", options.num_lookaheads);
		  printf("\t-i <int>: iter. refinement   (default %4d)\n", options.IterRefine);
		  exit(0);
		  break;
	      case 'r': nprow_A = atoi(*cpp);
		        break;
	      case 'c': npcol_A = atoi(*cpp);
		        break;
          case 'w': nprow_B = atoi(*cpp);
                break;
          case 'v': npcol_B = atoi(*cpp);
                break;
          case 'l': lookahead = atoi(*cpp);
                    break;
          case 'p': rowperm = atoi(*cpp);
                    break;
          case 'q': colperm = atoi(*cpp);
                    break;
	      case 's': symbfact = atoi(*cpp);
        	        break;
          case 'i': ir = atoi(*cpp);
                    break;
          case 'f': strcpy(postfix, *cpp);
                    break;
	    }
	} else { /* Last arg is considered a filename */
	    char buf_A[2];
        buf_A[0] = 'A'; buf_A[1] = '\0';
        char name_A[100];
        strcpy(name_A, *cpp);
        strcat(name_A, buf_A);
        strcat(name_A, postfix);
        if (!(fp_A = fopen (name_A, "r")))
        {
            ABORT ("File for A does not exist");
        }

        char buf_B[2];
        buf_B[0] = 'B'; buf_B[1] = '\0';
        char name_B[100];
        strcpy(name_B, *cpp);
        strcat(name_B, buf_B);
        strcat(name_B, postfix);
        if (!(fp_B = fopen (name_B, "r")))
        {
            ABORT ("File for B does not exist");
        }

        char buf_shift[2];
        buf_shift[0] = 'I'; buf_shift[1] = '\0';
        char name_shift[100];
        strcpy(name_shift, *cpp);
        strcat(name_shift, buf_shift);
        strcat(name_shift, postfix);
        if (!(fp_shift = fopen (name_shift, "r")))
        {
            ABORT ("File for shift intervals does not exist");
        }

        char buf_F[2];
        buf_F[0] = 'F'; buf_F[1] = '\0';
        char name_F[100];
        strcpy(name_F, *cpp);
        strcat(name_F, buf_F);
        strcat(name_F, postfix);
        if (!(fp_F = fopen (name_F, "r")))
        {
            ABORT ("File for RHS does not exist");
        }

        char buf_X[2];
        buf_X[0] = 'X'; buf_X[1] = '\0';
        char name_X[100];
        strcpy(name_X, *cpp);
        strcat(name_X, buf_X);
        strcat(name_X, postfix);
        if (!(fp_X = fopen (name_X, "r")))
        {
            ABORT ("File for true solution does not exist");
        }
	    break;
	}
    }

    /* Command line input to modify default options */
    if (rowperm != -1) options.RowPerm = rowperm;
    if (colperm != -1) options.ColPerm = colperm;
    if (lookahead != -1) options.num_lookaheads = lookahead;
    if (ir != -1) options.IterRefine = ir;
    if (symbfact != -1) options.ParSymbFact = symbfact;

    adi_gridinit_matrix(MPI_COMM_WORLD, nprow_A, npcol_A, &grid_A, nprow_B, npcol_B, &grid_B);

    if (grid_A.iam==0) {
        printf("Successfully generate two grids.\n");
        fflush(stdout);
    }
    // int global_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    // printf("Global rank is %d, id in grid_A is %d, id in grid_B is %d, it has A comm %d, and it has B comm %d.\n", 
    //     global_rank, grid_A.iam, grid_B.iam, grid_A.comm != MPI_COMM_NULL, grid_B.comm != MPI_COMM_NULL);
    // fflush(stdout);
    // char comm_name_A[MPI_MAX_OBJECT_NAME];
    // int name_length_A;
    // char comm_name_B[MPI_MAX_OBJECT_NAME];
    // int name_length_B;

    // if (grid_A.iam != -1) {
    //     MPI_Comm_get_name(grid_A.comm, comm_name_A, &name_length_A);
    //     printf("Global rank is %d, id in grid_A is %d with comm %s, in grid_B is %d.\n", 
    //         global_rank, grid_A.iam, comm_name_A, grid_B.iam);
    // }
    // if (grid_B.iam != -1) {
    //     MPI_Comm_get_name(grid_B.comm, comm_name_B, &name_length_B);
    //     printf("Global rank is %d, id in grid_B is %d with comm %s, in grid_A is %d.\n", 
    //         global_rank, grid_B.iam, comm_name_B, grid_A.iam);
    // }
    // if ((grid_A.iam != -1) && (grid_B.iam != -1)) {
    //     printf("Process %d appears in both grids! Wrong!\n", global_rank);
    // }
    // fflush(stdout);
    
    if (grid_A.iam==0) {
	    MPI_Query_thread(&omp_mpi_level);
        switch (omp_mpi_level) {
          case MPI_THREAD_SINGLE:
		printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
		fflush(stdout);
	        break;
          case MPI_THREAD_FUNNELED:
		printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
		fflush(stdout);
	        break;
          case MPI_THREAD_SERIALIZED:
		printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
		fflush(stdout);
	        break;
          case MPI_THREAD_MULTIPLE:
		printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
		fflush(stdout);
	        break;
        }
    }

    /* Bail out if I do not belong in the grid. */
    iam_A = grid_A.iam;
    iam_B = grid_B.iam;
    if (( (iam_A >= nprow_A * npcol_A) || (iam_A == -1) ) &&  ((iam_B >= nprow_B * npcol_B) || (iam_B == -1) )) goto out;
    if ( (!iam_A) ) {
    	int v_major, v_minor, v_bugfix;

    	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

    	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
    	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

    	// printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid for A:\t\t%d X %d\n", (int)grid_A.nprow, (int)grid_A.npcol);
        printf("Process grid for B:\t\t%d X %d\n", (int)grid_B.nprow, (int)grid_B.npcol);
    	fflush(stdout);
    }

    /* print solver options */
    if ( (!iam_A) ) {
    	print_options_dist(&options);
    	fflush(stdout);
    }

    suffix = &(postfix[1]);
    // printf("%s\n", postfix);

    /* ------------------------------------------------------------
       GET THE RIGHT HAND SIDE, SHIFT PARAMETERS, AND TRUE SOLUTION FROM FILE.
       ------------------------------------------------------------*/
    dread_shift_interval(fp_shift, &i1, &i2, &i3, &i4, &grid_A);
    dgenerate_shifts(i1, i2, i3, i4, &pp, &qq, &ll, &grid_A, &grid_B);
    // printf("id in grid_A is %d, id in grid_B is %d, it gets shifts with length %d with first element %f and %f.\n", 
    //     grid_A.iam, grid_B.iam, ll, pp[0], qq[0]);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);
    if (!iam_A) {
        // printf("Get shifts!\n");
        printf("Shifts found are\n");
        printf("p: ");
        for (int j = 0; j < ll; ++j) {
            printf("%f ", pp[j]);
        }
        printf("\n");
        printf("q: ");
        for (int j = 0; j < ll; ++j) {
            printf("%f ", qq[j]);
        }
        printf("\n");
        fflush(stdout);
    }

    dread_RHS(fp_F, &F, &F_transpose, &grid_A, &grid_B, &m_A, &m_B, &ldf, &ldft, &F_global);
    if (!iam_A) {
        printf("Read RHS!\n");
        fflush(stdout);
    }
    // printf("id in grid_A is %d, id in grid_B is %d, gets local RHS.\n", 
    //     grid_A.iam, grid_B.iam);
    // if (iam_A != -1) {
    //     printf("First element of F is %f on id %d in grid_A", F[0], iam_A);
    //     fflush(stdout);
    // }
    // if (iam_B != -1) {
    //     printf("First element of F_transpose is %f on id %d in grid_B", F_transpose[0], iam_B);
    //     fflush(stdout);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // fflush(stdout);

    dread_X(fp_X, &trueX_global, &grid_A);
    if (!iam_A) {
        printf("Read true solution!\n");
        // printf("First element is %f.\n", trueX_global[0]);
        fflush(stdout);
    }
    // printf("id in grid_A is %d, id in grid_B is %d, gets to true solution location.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);

    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       GET A and B FROM FILE.
       ------------------------------------------------------------*/
    int_t m_Atmp, n_Atmp;
    int_t nnz_A;
    double *nzval_A;
    int_t *rowind_A, *colptr_A;
    dread_matrix(fp_A, suffix, &grid_A, &m_Atmp, &n_Atmp, &nnz_A, &nzval_A, &rowind_A, &colptr_A);

    if (!iam_A) {
        printf("Read A!\n");
        fflush(stdout);
    }
    // printf("id in grid_A is %d, id in grid_B is %d, gets to A matrix location.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    int_t m_Btmp, n_Btmp;
    int_t nnz_B;
    double *nzval_B;
    int_t *rowind_B, *colptr_B;
    dread_matrix(fp_B, suffix, &grid_B, &m_Btmp, &n_Btmp, &nnz_B, &nzval_B, &rowind_B, &colptr_B);

    if (!iam_B) {
        printf("Read B!\n");
        fflush(stdout);
    }

    // printf("id in grid_A is %d, id in grid_B is %d, gets all data ready.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       NOW WE SOLVE THE ADI.
       ------------------------------------------------------------*/
    if ( !(X = doubleMalloc_dist(m_A*m_B)) )
        ABORT("Malloc fails for X.");

    adi(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        &grid_A, &grid_B, F, ldf, F_transpose, ldft, pp, qq, ll, X);

    if (!iam_A) {
        printf("Get solution!\n");
        fflush(stdout);
    }

    // printf("id in grid_A is %d, id in grid_B is %d, finishes solving.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       CHECK ACCURACY OF REPRODUCED RHS.
       ------------------------------------------------------------*/
    dcheck_error(m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        &grid_A, &grid_B, F_global, X, trueX_global);

    // printf("id in grid_A is %d, id in grid_B is %d, finishes checking errors.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    SUPERLU_FREE(X);
    if (iam_A != -1) {
        SUPERLU_FREE(F);
        SUPERLU_FREE(F_global);
    }
    if (iam_B != -1) {
        SUPERLU_FREE(F_transpose);
    }
    SUPERLU_FREE(pp);
    SUPERLU_FREE(qq);
    if (!iam_A) {
        SUPERLU_FREE(trueX_global);
    }

    fclose(fp_A);
    fclose(fp_B);
    fclose(fp_shift);
    fclose(fp_F);
    fclose(fp_X);

    // printf("id in grid_A is %d, id in grid_B is %d, releases memory.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    adi_gridexit_matrix(&grid_A, &grid_B);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

}
