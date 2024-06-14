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
    gridinfo_t grid_A, grid_B, grid_C, grid_D;
    gridinfo_t **grids;
    int    m_A, m_B, m_C, m_D, r1, r2, r3, rank1, rank2, rank3, mm, rr, dim;
    int    ms[4], locals[4], rs[3], nrhss[3];
    int.   nprow_all, npcol_all;
    int    nprow_A, npcol_A, nprow_B, npcol_B, nprow_C, npcol_C, nprow_D, npcol_D, lookahead, colperm, rowperm, ir, symbfact;
    int    nprows[4], npcols[4], grid_proc[4];
    int    iam_A, iam_B, iam_C, iam_D, info, ldu1, ldu2, ldu3, ldv, ldt1, ldt2, ldt3, ldt4;
    char   **cpp, c, *suffix;
    char   postfix[10];
    FILE   *fp_A, *fp_B, *fp_C, *fp_D, *fp_shift1, *fp_shift2, *fp_shift3;
    FILE   *fp_int, *fp_U1, *fp_U2, *fp_U3, *fp_V, *fp_X, *fp_F, *fp_M, *fp_R, *fopen();
    int ii, omp_mpi_level;
    int p; /* The following variables are used for batch solves */
    int_t i, j;
    double *pps[3], *qqs[3];
    double *V, *trueX_global, *U1_global, *U2_global, *U3_global, *V_global, *trueF_global;
    double *Us[3];
    double tol = 1.0/(1e10);
    double las[2], uas[2], lbs[2], ubs[2];
    int_t lls[3];
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;

    dim = 4;
    nprow_A = 1;  /* Default process rows.      */
    npcol_A = 1;  /* Default process columns.   */
    nprow_B = 1;  /* Default process rows.      */
    npcol_B = 1;  /* Default process columns.   */
    nprow_C = 1;  /* Default process rows.      */
    npcol_C = 1;  /* Default process columns.   */
    nprow_D = 1;  /* Default process rows.      */
    npcol_D = 1;  /* Default process columns.   */
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
		  printf("\t-a <int>: process rows for all grids     (default %4d)\n", nprow_all);
		  printf("\t-b <int>: process columns for all grids  (default %4d)\n", npcol_all);
		  printf("\t-p <int>: row permutation    (default %4d)\n", options.RowPerm);
		  printf("\t-q <int>: col permutation    (default %4d)\n", options.ColPerm);
		  printf("\t-s <int>: parallel symbolic? (default %4d)\n", options.ParSymbFact);
		  printf("\t-l <int>: lookahead level    (default %4d)\n", options.num_lookaheads);
		  printf("\t-i <int>: iter. refinement   (default %4d)\n", options.IterRefine);
		  exit(0);
		  break;
	      case 'a': nprow_all = atoi(*cpp);
		        break;
	      case 'b': npcol_all = atoi(*cpp);
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

        char buf_C[2];
        buf_C[0] = 'C'; buf_C[1] = '\0';
        char name_C[100];
        strcpy(name_C, *cpp);
        strcat(name_C, buf_C);
        strcat(name_C, postfix);
        if (!(fp_C = fopen (name_C, "r")))
        {
            ABORT ("File for C does not exist");
        }

        char buf_D[2];
        buf_D[0] = 'D'; buf_D[1] = '\0';
        char name_D[100];
        strcpy(name_D, *cpp);
        strcat(name_D, buf_D);
        strcat(name_D, postfix);
        if (!(fp_D = fopen (name_D, "r")))
        {
            ABORT ("File for D does not exist");
        }

        char buf_shift1[3];
        buf_shift1[0] = 'S'; buf_shift1[1] = 'a'; buf_shift1[2] = '\0';
        char name_shift1[100];
        strcpy(name_shift1, *cpp);
        strcat(name_shift1, buf_shift1);
        strcat(name_shift1, postfix);
        if (!(fp_shift1 = fopen (name_shift1, "r")))
        {
            ABORT ("File for first sets of shifts does not exist");
        }

        char buf_shift2[3];
        buf_shift2[0] = 'S'; buf_shift2[1] = 'b'; buf_shift2[2] = '\0';
        char name_shift2[100];
        strcpy(name_shift2, *cpp);
        strcat(name_shift2, buf_shift2);
        strcat(name_shift2, postfix);
        if (!(fp_shift2 = fopen (name_shift2, "r")))
        {
            ABORT ("File for second sets of shifts does not exist");
        }

        char buf_shift3[3];
        buf_shift3[0] = 'S'; buf_shift3[1] = 'c'; buf_shift3[2] = '\0';
        char name_shift3[100];
        strcpy(name_shift3, *cpp);
        strcat(name_shift3, buf_shift3);
        strcat(name_shift3, postfix);
        if (!(fp_shift3 = fopen (name_shift3, "r")))
        {
            ABORT ("File for third sets of shifts does not exist");
        }

        char buf_int[2];
        buf_int[0] = 'I'; buf_int[1] = '\0';
        char name_I[100];
        strcpy(name_I, *cpp);
        strcat(name_I, buf_int);
        strcat(name_I, postfix);
        if (!(fp_int = fopen (name_I, "r")))
        {
            ABORT ("File for intervals of spectra of matrices does not exist");
        }

        char buf_U1[3];
        buf_U1[0] = 'U'; buf_U1[1] = 'a'; buf_U1[2] = '\0';
        char name_U1[100];
        strcpy(name_U1, *cpp);
        strcat(name_U1, buf_U1);
        strcat(name_U1, postfix);
        if (!(fp_U1 = fopen (name_U1, "r")))
        {
            ABORT ("File for RHS first factor of first unfolding does not exist");
        }

        char buf_U2[3];
        buf_U2[0] = 'U'; buf_U2[1] = 'b'; buf_U2[2] = '\0';
        char name_U2[100];
        strcpy(name_U2, *cpp);
        strcat(name_U2, buf_U2);
        strcat(name_U2, postfix);
        if (!(fp_U2 = fopen (name_U2, "r")))
        {
            ABORT ("File for RHS first factor of second unfolding does not exist");
        }

        char buf_U3[3];
        buf_U3[0] = 'U'; buf_U3[1] = 'c'; buf_U3[2] = '\0';
        char name_U3[100];
        strcpy(name_U3, *cpp);
        strcat(name_U3, buf_U3);
        strcat(name_U3, postfix);
        if (!(fp_U3 = fopen (name_U3, "r")))
        {
            ABORT ("File for RHS first factor of third unfolding does not exist");
        }

        char buf_V[2];
        buf_V[0] = 'V'; buf_V[1] = '\0';
        char name_V[100];
        strcpy(name_V, *cpp);
        strcat(name_V, buf_V);
        strcat(name_V, postfix);
        if (!(fp_V = fopen (name_V, "r")))
        {
            ABORT ("File for RHS second factor of third unfolding does not exist");
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

        char buf_F[2];
        buf_F[0] = 'F'; buf_F[1] = '\0';
        char name_F[100];
        strcpy(name_F, *cpp);
        strcat(name_F, buf_F);
        strcat(name_F, postfix);
        if (!(fp_F = fopen (name_F, "r")))
        {
            ABORT ("File for true solution does not exist");
        }

        char buf_M[2];
        buf_M[0] = 'M'; buf_M[1] = '\0';
        char name_M[100];
        strcpy(name_M, *cpp);
        strcat(name_M, buf_M);
        strcat(name_M, postfix);
        if (!(fp_M = fopen (name_M, "r")))
        {
            ABORT ("File for size of problem does not exist");
        }

        char buf_R[2];
        buf_R[0] = 'R'; buf_R[1] = '\0';
        char name_R[100];
        strcpy(name_R, *cpp);
        strcat(name_R, buf_R);
        strcat(name_R, postfix);
        if (!(fp_R = fopen (name_R, "r")))
        {
            ABORT ("File for TT rank of RHS does not exist");
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

    nprow_A = nprow_all; npcol_A = npcol_all;
    nprow_B = nprow_all; npcol_B = npcol_all;
    nprow_C = nprow_all; npcol_C = npcol_all;
    nprow_D = nprow_all; npcol_D = npcol_all;
    nprows[0] = nprow_A; nprows[1] = nprow_B; nprows[2] = nprow_C; nprows[3] = nprow_D;
    npcols[0] = npcol_A; npcols[1] = npcol_B; npcols[2] = npcol_C; npcols[3] = npcol_D;
    grid_proc[0] = nprow_A * npcol_A; grid_proc[1] = nprow_B * npcol_B; 
    grid_proc[2] = nprow_C * npcol_C; grid_proc[3] = nprow_D * npcol_D;
    grids = SUPERLU_MALLOC(4*sizeof(gridinfo_t*));
    grids[0] = &grid_A; grids[1] = &grid_B; grids[2] = &grid_C; grids[3] = &grid_D;
    adi_gridinit_tensor(MPI_COMM_WORLD, dim, nprows, npcols, grid_proc, grids);

    if (grid_A.iam==0) {
        printf("Successfully generate %d grids.\n", dim);
        fflush(stdout);
    }
    
    // printf("Global rank is %d, id in grid_A is %d, id in grid_B is %d, it has A comm %d, and it has B comm %d.\n", 
    //     global_rank, grid_A.iam, grid_B.iam, grid_A.comm != MPI_COMM_NULL, grid_B.comm != MPI_COMM_NULL);
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
    iam_C = grid_C.iam;
    iam_D = grid_D.iam;
    if (( (iam_A >= nprow_A * npcol_A) || (iam_A == -1) ) && ((iam_B >= nprow_B * npcol_B) || (iam_B == -1) ) 
        && ( (iam_C >= nprow_C * npcol_C) || (iam_C == -1) ) && ( (iam_D >= nprow_D * npcol_D) || (iam_D == -1) )) 
        goto out;
    if ( (!iam_A) ) {
    	int v_major, v_minor, v_bugfix;

    	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

    	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
    	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

    	// printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid for A:\t\t%d X %d\n", (int)grid_A.nprow, (int)grid_A.npcol);
        printf("Process grid for B:\t\t%d X %d\n", (int)grid_B.nprow, (int)grid_B.npcol);
        printf("Process grid for C:\t\t%d X %d\n", (int)grid_C.nprow, (int)grid_C.npcol);
        printf("Process grid for D:\t\t%d X %d\n", (int)grid_D.nprow, (int)grid_D.npcol);
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
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    dread_size(fp_M, dim, ms, grids);
    dread_size(fp_R, dim-1, nrhss, grids);
    m_A = ms[0]; m_B = ms[1]; m_C = ms[2]; m_D = ms[3];
    r1 = nrhss[0]; r2 = nrhss[1]; r3 = nrhss[2];

    // printf("Process with id %d in grid_A, %d in grid_B, %d in grid_C, and %d in grid_D gets problem size %d, %d, %d, %d, and rhs rank %d, %d, %d.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam, grid_D.iam, ms[0], ms[1], ms[2], ms[3], nrhss[0], nrhss[1], nrhss[2]);
    // fflush(stdout);

    if (iam_A != -1) {
        dread_shift_onegrid(fp_shift1, &(pps[0]), &(qqs[0]), &(lls[0]), &grid_A);

        // printf("Grid_A proc %d gets first elements of first shift with length %d to be %f and %f.\n", 
        //     iam_A, lls[0], pps[0][0], qqs[0][0]);
        // fflush(stdout);

        if (!iam_A) {
            printf("Read shifts for first equation!\n");
            fflush(stdout);
        }
    }

    if (iam_B != -1) {
        dread_shift_onegrid(fp_shift2, &(pps[1]), &(qqs[1]), &(lls[1]), &grid_B);

        // printf("Grid_B proc %d gets first elements of first shift with length %d to be %f and %f.\n", 
        //     iam_B, lls[1], pps[1][0], qqs[1][0]);
        // fflush(stdout);

        if (!iam_B) {
            printf("Read shifts for second equation!\n");
            fflush(stdout);
        }
    }

    if ((iam_C != -1) || (iam_D != -1)) {
        dread_shift_twogrids(fp_shift3, &(pps[2]), &(qqs[2]), &(lls[2]), &grid_C, &grid_D, 2, 3, grid_proc);

        // printf("Proc %d in grid_C and %d in grid_D gets first elements of second shift with length %d to be %f and %f.\n", 
        //     iam_C, iam_D, lls[2], pps[2][0], qqs[2][0]);
        // fflush(stdout);

        if (!iam_C) {
            printf("Read shifts for third equation!\n");
            fflush(stdout);
        }
    }

    dread_shift_interval_multigrids(fp_int, dim, las, uas, lbs, ubs, grids, grid_proc);
    if (!iam_A) {
        // printf("Read spectra bound of A and B to be %f, %f, %f, %f!\n", las[0], uas[0], lbs[0], ubs[0]);
        printf("Read spectra bound of matrix combinations!\n");
        fflush(stdout);
    }
    
    if (iam_A != -1) {
        dread_RHS_factor(fp_U1, &(Us[0]), &grid_A, &mm, &rr, &(locals[0]), &U1_global);
        if (!iam_A) {
            printf("Read U1!\n");
            fflush(stdout);
        }
        // printf("Grid_A proc %d gets first and last element of U1 to be %f and %f.\n", iam_A, Us[0][0], Us[0][locals[0]*r1-1]);
        // fflush(stdout);
    }

    if (iam_B != -1) {
        dread_RHS_factor_multidim(fp_U2, &(Us[1]), &grid_B, ms, 1, &(locals[1]), &U2_global);
        if (!iam_B) {
            printf("Read U2!\n");
            fflush(stdout);
        }
        // printf("Grid_B proc %d gets first and last element of U2 to be %f and %f.\n", iam_B, Us[1][0], Us[1][m_A*locals[1]*r2-1]);
        // fflush(stdout);
    }

    if (iam_C != -1) {
        dread_RHS_factor_multidim(fp_U3, &(Us[2]), &grid_C, ms, 2, &(locals[2]), &U3_global);
        if (!iam_C) {
            printf("Read U3!\n");
            fflush(stdout);
        }
        // printf("Grid_C proc %d gets first and last element of U3 to be %f and %f.\n", iam_C, Us[2][0], Us[2][m_A*m_B*locals[2]*r3-1]);
        // fflush(stdout);
    }

    if (iam_D != -1) {
        dread_RHS_factor(fp_V, &V, &grid_D, &mm, &rr, &(locals[3]), &V_global);
        if (!iam_D) {
            printf("Read V!\n");
            fflush(stdout);
        }
        // printf("Grid_D proc %d gets first and last element of V to be %f and %f.\n", iam_D, V[0], V[locals[3]*r3-1]);
        // fflush(stdout);
    }

    dread_X(fp_X, &trueX_global, &grid_A);
    if (!iam_A) {
        printf("Read true solution!\n");
        // printf("First element is %f.\n", trueX_global[0]);
        fflush(stdout);
    }

    dread_X(fp_F, &trueF_global, &grid_A);
    if (!iam_A) {
        printf("Read global RHS!\n");
        // printf("First element is %f.\n", trueX_global[0]);
        fflush(stdout);
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       GET A and B FROM FILE.
       ------------------------------------------------------------*/
    int_t nnzs[4];
    double* nzvals[4];
    int_t* rowinds[4];
    int_t* colptrs[4];

    int_t m_Atmp, n_Atmp;
    // int_t nnz_A;
    dread_matrix(fp_A, suffix, &grid_A, &m_Atmp, &n_Atmp, &(nnzs[0]), &(nzvals[0]), &(rowinds[0]), &(colptrs[0]));
    
    // double *nzval_A;
    // int_t *rowind_A, *colptr_A;
    if (!iam_A) {
        printf("Read A!\n");
        fflush(stdout);
    }
    // if (iam_A != -1) {
    //     dallocateA_dist(m_A, nnz_A, &nzval_A, &rowind_A, &colptr_A);
    //     for (i = 0; i < m_A; ++i) {
    //         for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
    //             nzval_A[j] = nzvals[0][j];
    //             rowind_A[j] = rowinds[0][j];
    //         }
    //         colptr_A[i] = colptrs[0][i];
    //     }
    //     colptr_A[m_A] = colptrs[0][m_A];
    // }

    int_t m_Btmp, n_Btmp;
    // int_t nnz_B;
    dread_matrix(fp_B, suffix, &grid_B, &m_Btmp, &n_Btmp, &(nnzs[1]), &(nzvals[1]), &(rowinds[1]), &(colptrs[1]));

    // double *nzval_B;
    // int_t *rowind_B, *colptr_B;
    if (!iam_B) {
        printf("Read B!\n");
        fflush(stdout);
    }
    // if (iam_B != -1) {
    //     dallocateA_dist(m_B, nnz_B, &nzval_B, &rowind_B, &colptr_B);
    //     for (i = 0; i < m_B; ++i) {
    //         for (j = colptrs[1][i]; j < colptrs[1][i+1]; ++j) {
    //             nzval_B[j] = nzvals[1][j];
    //             rowind_B[j] = rowinds[1][j];
    //         }
    //         colptr_B[i] = colptrs[1][i];
    //     }
    //     colptr_B[m_B] = colptrs[1][m_B];
    // }

    int_t m_Ctmp, n_Ctmp;
    // int_t nnz_C;
    dread_matrix(fp_C, suffix, &grid_C, &m_Ctmp, &n_Ctmp, &(nnzs[2]), &(nzvals[2]), &(rowinds[2]), &(colptrs[2]));

    // double *nzval_C;
    // int_t *rowind_C, *colptr_C;
    if (!iam_C) {
        printf("Read C!\n");
        fflush(stdout);
    }
    // if (iam_C != -1) {
    //     dallocateA_dist(m_C, nnz_C, &nzval_C, &rowind_C, &colptr_C);
    //     for (i = 0; i < m_C; ++i) {
    //         for (j = colptrs[2][i]; j < colptrs[2][i+1]; ++j) {
    //             nzval_C[j] = nzvals[2][j];
    //             rowind_C[j] = rowinds[2][j];
    //         }
    //         colptr_C[i] = colptrs[2][i];
    //     }
    //     colptr_C[m_C] = colptrs[2][m_C];
    // }

    int_t m_Dtmp, n_Dtmp;
    // int_t nnz_D;
    dread_matrix(fp_D, suffix, &grid_D, &m_Dtmp, &n_Dtmp, &(nnzs[3]), &(nzvals[3]), &(rowinds[3]), &(colptrs[3]));

    // double *nzval_D;
    // int_t *rowind_D, *colptr_D;
    if (!iam_D) {
        printf("Read D!\n");
        fflush(stdout);
    }
    // if (iam_D != -1) {
    //     dallocateA_dist(m_D, nnz_D, &nzval_D, &rowind_D, &colptr_D);
    //     for (i = 0; i < m_D; ++i) {
    //         for (j = colptrs[3][i]; j < colptrs[3][i+1]; ++j) {
    //             nzval_D[j] = nzvals[3][j];
    //             rowind_D[j] = rowinds[3][j];
    //         }
    //         colptr_D[i] = colptrs[3][i];
    //     }
    //     colptr_D[m_D] = colptrs[3][m_D];
    // }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       NOW WE SOLVE THE ADI.
       ------------------------------------------------------------*/
    double* TTcores[4];
    fadi_ttsvd(options, dim, ms, nnzs, nzvals, rowinds, colptrs, grids, Us, V, locals, nrhss, pps, qqs, lls, tol,
        las, uas, lbs, ubs, TTcores, rs, grid_proc);
    
    if (!iam_A) {
        printf("Get solution!\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // printf("Process with id %d in grid_A, %d in grid_B, %d in grid_C, and %d in grid_D finishes solving.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam, grid_D.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       CHECK ACCURACY OF REPRODUCED RHS.
       ------------------------------------------------------------*/
    
    dcheck_error_TT(ms, nnzs, nzvals, rowinds, colptrs, grids, rs, locals, dim, trueF_global, TTcores, trueX_global, grid_proc);

    // printf("Process with id %d in grid_A, %d in grid_B, %d in grid_C, and %d in grid_D finishes checking errors.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam, grid_D.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    if (iam_A != -1) {
        SUPERLU_FREE(Us[0]);
        SUPERLU_FREE(U1_global);
        SUPERLU_FREE(pps[0]);
        SUPERLU_FREE(qqs[0]);
        SUPERLU_FREE(TTcores[0]);
    }
    if (iam_B != -1) {
        SUPERLU_FREE(Us[1]);
        SUPERLU_FREE(U2_global);
        SUPERLU_FREE(pps[1]);
        SUPERLU_FREE(qqs[1]);
        SUPERLU_FREE(TTcores[1]);
    }
    if (iam_C != -1) {
        SUPERLU_FREE(Us[2]);
        SUPERLU_FREE(U3_global);
        SUPERLU_FREE(pps[2]);
        SUPERLU_FREE(qqs[2]);
        SUPERLU_FREE(TTcores[2]);
    }
    if (iam_D != -1) {
        SUPERLU_FREE(V);
        SUPERLU_FREE(V_global);
        SUPERLU_FREE(pps[2]);
        SUPERLU_FREE(qqs[2]);
        SUPERLU_FREE(TTcores[3]);
    }
    
    if (!iam_A) {
        SUPERLU_FREE(trueF_global);
        SUPERLU_FREE(trueX_global);
    }

    fclose(fp_A);
    fclose(fp_B);
    fclose(fp_C);
    fclose(fp_D);
    fclose(fp_shift1);
    fclose(fp_shift2);
    fclose(fp_shift3);
    fclose(fp_int);
    fclose(fp_U1);
    fclose(fp_U2);
    fclose(fp_U3);
    fclose(fp_V);
    fclose(fp_F);
    fclose(fp_X);
    fclose(fp_M);
    fclose(fp_R);

    // printf("Process with id %d in grid_A, %d in grid_B, %d in grid_C, and %d in grid_D releases memory.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam, grid_D.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    adi_gridexit_tensor(grids, dim);
    SUPERLU_FREE(grids);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();
}
