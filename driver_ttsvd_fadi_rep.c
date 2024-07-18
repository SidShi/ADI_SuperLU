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
    gridinfo_t grid1, grid2;
    gridinfo_t **grids;
    int    m_A, m_B, m_C, r1, r2, rank1, rank2, mm, rr;
    int    ms[3], rs[2];
    int    nprow1, npcol1, nprow2, npcol2, lookahead, colperm, rowperm, ir, symbfact, rep;
    int    nprows[2], npcols[2], grid_proc[2];
    int    iam1, iam2, info, ldu1, ldu2, ldv1, ldv2, ldt1, ldt2, ldt3;
    char   **cpp, c, *suffix;
    char   postfix[10];
    FILE   *fp_A, *fp_B, *fp_C, *fp_shift1, *fp_shift2, *fp_int, *fp_U1, *fp_U2, *fp_V1, *fp_V2, *fp_X, *fp_F, *fp_M, *fp_R, *fopen();
    int ii, omp_mpi_level;
    int p; /* The following variables are used for batch solves */
    int_t i, j;
    double *pp1, *qq1, *pp2, *qq2;
    double *U1, *U2, *V1, *V2, *trueX_global, *U1_global, *U2_global, *V1_global, *V2_global, *trueF_global;
    double tol = 1.0/(1e10);
    double la[2], ua[2], lb[2], ub[2];
    int_t ll1, ll2;
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;

    nprow1 = 1;  /* Default process rows.      */
    npcol1 = 1;  /* Default process columns.   */
    nprow2 = 1;  /* Default process rows.      */
    npcol2 = 1;  /* Default process columns.   */
    rep = 0;
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
		  printf("\t-r <int>: process rows grid 1     (default %4d)\n", nprow1);
		  printf("\t-c <int>: process columns grid 1  (default %4d)\n", npcol1);
          printf("\t-w <int>: process rows grid 2     (default %4d)\n", nprow2);
          printf("\t-v <int>: process columns grid 2  (default %4d)\n", npcol2);
          printf("\t-t <int>: TT fADI refinement iteration  (default %4d)\n", rep);
          printf("\t-p <int>: row permutation    (default %4d)\n", options.RowPerm);
		  printf("\t-q <int>: col permutation    (default %4d)\n", options.ColPerm);
		  printf("\t-s <int>: parallel symbolic? (default %4d)\n", options.ParSymbFact);
		  printf("\t-l <int>: lookahead level    (default %4d)\n", options.num_lookaheads);
		  printf("\t-i <int>: iter. refinement   (default %4d)\n", options.IterRefine);
		  exit(0);
		  break;
	      case 'r': nprow1 = atoi(*cpp);
		        break;
	      case 'c': npcol1 = atoi(*cpp);
		        break;
          case 'w': nprow2 = atoi(*cpp);
                break;
          case 'v': npcol2 = atoi(*cpp);
                break;
          case 't': rep = atoi(*cpp);
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

        char buf_int[2];
        buf_int[0] = 'I'; buf_int[1] = '\0';
        char name_I[100];
        strcpy(name_I, *cpp);
        strcat(name_I, buf_int);
        strcat(name_I, postfix);
        if (!(fp_int = fopen (name_I, "r")))
        {
            ABORT ("File for intervals of spectra of A, B, and C does not exist");
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

        char buf_V1[3];
        buf_V1[0] = 'V'; buf_V1[1] = 'a'; buf_V1[2] = '\0';
        char name_V1[100];
        strcpy(name_V1, *cpp);
        strcat(name_V1, buf_V1);
        strcat(name_V1, postfix);
        if (!(fp_V1 = fopen (name_V1, "r")))
        {
            ABORT ("File for RHS first factor of second unfolding does not exist");
        }

        char buf_V2[3];
        buf_V2[0] = 'V'; buf_V2[1] = 'b'; buf_V2[2] = '\0';
        char name_V2[100];
        strcpy(name_V2, *cpp);
        strcat(name_V2, buf_V2);
        strcat(name_V2, postfix);
        if (!(fp_V2 = fopen (name_V2, "r")))
        {
            ABORT ("File for RHS second factor of second unfolding does not exist");
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

    nprows[0] = nprow1; nprows[1] = nprow2;
    npcols[0] = npcol1; npcols[1] = npcol2;
    grid_proc[0] = nprow1 * npcol1; grid_proc[1] = nprow2 * npcol2;
    grids = SUPERLU_MALLOC(2*sizeof(gridinfo_t*));
    grids[0] = &grid1; grids[1] = &grid2;
    adi_gridinit_matrix(MPI_COMM_WORLD, nprow1, npcol1, &grid1, nprow2, npcol2, &grid2);

    if (grid1.iam==0) {
        printf("Successfully generate 2 grids.\n");
        fflush(stdout);
    }
    
    // printf("Global rank is %d, id in grid_A is %d, id in grid_B is %d, it has A comm %d, and it has B comm %d.\n", 
    //     global_rank, grid_A.iam, grid_B.iam, grid_A.comm != MPI_COMM_NULL, grid_B.comm != MPI_COMM_NULL);
    // fflush(stdout);
    
    if (grid1.iam==0) {
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
    iam1 = grid1.iam;
    iam2 = grid2.iam;
    if (( (iam1 >= nprow1 * npcol1) || (iam1 == -1) ) && ((iam2 >= nprow2 * npcol2) || (iam2 == -1) )) 
        goto out;
    if ( (!iam1) ) {
    	int v_major, v_minor, v_bugfix;

    	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

    	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
    	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

    	// printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid 1:\t\t%d X %d\n", (int)grid1.nprow, (int)grid1.npcol);
        printf("Process grid 2:\t\t%d X %d\n", (int)grid2.nprow, (int)grid2.npcol);
    	fflush(stdout);
    }

    /* print solver options */
    if ( (!iam1) ) {
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

    dread_size(fp_M, 3, ms, grids);
    dread_size(fp_R, 2, rs, grids);
    m_A = ms[0]; m_B = ms[1]; m_C = ms[2];
    r1 = rs[0]; r2 = rs[1];

    // printf("Process with id in grid_A %d, id in grid_B %d, and id in grid_C %d gets problem size %d, %d, %d, and rhs rank %d, %d.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam, ms[0], ms[1], ms[2], rs[0], rs[1]);
    // fflush(stdout);

    if ((iam1 != -1) || (iam2 != -1)) {
        dread_shift_twogrids(fp_shift1, &pp1, &qq1, &ll1, &grid1, &grid2, 0, 1, grid_proc);
        dread_shift_twogrids(fp_shift2, &pp2, &qq2, &ll2, &grid1, &grid2, 0, 1, grid_proc);

        // printf("Proc %d in grid_B and %d in grid_C gets first elements of second shift with length %d to be %f and %f.\n", iam_B, iam_C, ll2, pp2[0], qq2[0]);
        // fflush(stdout);

        if (!iam1) {
            printf("Read shifts for first and second equation!\n");
            fflush(stdout);
        }

        dread_shift_multi_interval_2grids_2way(fp_int, 3, la, ua, lb, ub, &grid1, &grid2, grid_proc);

        if (!iam1) {
            // printf("Read spectra bound of A and B to be %f, %f, %f, %f!\n", la, ua, lb, ub);
            printf("Read spectra bound of A, B, and C!\n");
            fflush(stdout);
        }
    }

    if (iam1 != -1) {
        dread_RHS_factor(fp_U1, &U1, &grid1, &mm, &rr, &ldu1, &U1_global);
        if (!iam1) {
            printf("Read U1!\n");
            fflush(stdout);
        }
        // printf("Grid_A proc %d gets first and last element of U to be %f and %f.\n", iam_A, U[0], U[ldu*r-1]);
        // fflush(stdout);
    
        dread_RHS_factor_twodim(fp_U2, &U2, &grid1, m_A, &ldu2, &U2_global);
        if (!iam1) {
            printf("Read U2!\n");
            fflush(stdout);
        }
        // printf("Grid_B proc %d gets first and last element of V to be %f and %f.\n", iam_B, V[0], V[ldv*r-1]);
        // fflush(stdout);
    }

    if (iam2 != -1) {
        dread_RHS_factor_twodim(fp_V1, &V1, &grid2, m_C, &ldv1, &V1_global);
        if (!iam2) {
            printf("Read V1!\n");
            fflush(stdout);
        }
        // printf("Grid_B proc %d gets first and last element of V to be %f and %f.\n", iam_B, V[0], V[ldv*r-1]);
        // fflush(stdout);

        dread_RHS_factor(fp_V2, &V2, &grid2, &mm, &rr, &ldv2, &V2_global);
        if (!iam2) {
            printf("Read V2!\n");
            fflush(stdout);
        }
        // printf("Grid_B proc %d gets first and last element of V to be %f and %f.\n", iam_B, V[0], V[ldv*r-1]);
        // fflush(stdout);
    }

    if ((iam1 != -1) || (iam2 != -1)) {
        dread_X_twogrids(fp_X, &trueX_global, &grid1, &grid2, grid_proc);
        if (!iam1) {
            printf("Read true solution!\n");
            // printf("First element is %f.\n", trueX_global[0]);
            fflush(stdout);
        }

        dread_X_twogrids(fp_F, &trueF_global, &grid1, &grid2, grid_proc);
        if (!iam1) {
            printf("Read global RHS!\n");
            // printf("First element is %f.\n", trueX_global[0]);
            fflush(stdout);
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       GET A and B FROM FILE.
       ------------------------------------------------------------*/
    double* nzvals[3];
    int_t* rowinds[3];
    int_t* colptrs[3];

    int_t m_Atmp, n_Atmp;
    int_t nnz_A;
    dread_matrix(fp_A, suffix, &grid1, &m_Atmp, &n_Atmp, &nnz_A, &(nzvals[0]), &(rowinds[0]), &(colptrs[0]));
    
    double *nzval_A;
    int_t *rowind_A, *colptr_A;
    if (!iam1) {
        printf("Read A!\n");
        fflush(stdout);
    }
    if (iam1 != -1) {
        dallocateA_dist(m_A, nnz_A, &nzval_A, &rowind_A, &colptr_A);
        for (i = 0; i < m_A; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzval_A[j] = nzvals[0][j];
                rowind_A[j] = rowinds[0][j];
            }
            colptr_A[i] = colptrs[0][i];
        }
        colptr_A[m_A] = colptrs[0][m_A];
    }

    // printf("id in grid_A is %d, id in grid_B is %d, gets to A matrix location.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    int_t m_Btmp, n_Btmp;
    int_t nnz_B;
    dread_matrix_twogrids(fp_B, suffix, &grid1, &grid2, &m_Btmp, &n_Btmp, &nnz_B, &(nzvals[1]), &(rowinds[1]), &(colptrs[1]));

    double *nzval_B;
    int_t *rowind_B, *colptr_B;
    if (!iam1) {
        printf("Read B!\n");
        fflush(stdout);
    }
    if ((iam1 != -1) || (iam2 != -1)) {
        dallocateA_dist(m_B, nnz_B, &nzval_B, &rowind_B, &colptr_B);
        for (i = 0; i < m_B; ++i) {
            for (j = colptrs[1][i]; j < colptrs[1][i+1]; ++j) {
                nzval_B[j] = nzvals[1][j];
                rowind_B[j] = rowinds[1][j];
            }
            colptr_B[i] = colptrs[1][i];
        }
        colptr_B[m_B] = colptrs[1][m_B];
    }

    // printf("id in grid_A is %d, id in grid_B is %d, gets all data ready.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    int_t m_Ctmp, n_Ctmp;
    int_t nnz_C;
    dread_matrix(fp_C, suffix, &grid2, &m_Ctmp, &n_Ctmp, &nnz_C, &(nzvals[2]), &(rowinds[2]), &(colptrs[2]));

    double *nzval_C;
    int_t *rowind_C, *colptr_C;
    if (!iam2) {
        printf("Read C!\n");
        fflush(stdout);
    }
    if (iam2 != -1) {
        dallocateA_dist(m_C, nnz_C, &nzval_C, &rowind_C, &colptr_C);
        for (i = 0; i < m_C; ++i) {
            for (j = colptrs[2][i]; j < colptrs[2][i+1]; ++j) {
                nzval_C[j] = nzvals[2][j];
                rowind_C[j] = rowinds[2][j];
            }
            colptr_C[i] = colptrs[2][i];
        }
        colptr_C[m_C] = colptrs[2][m_C];
    }

    // printf("id in grid_A is %d, id in grid_B is %d, gets all data ready.\n", 
    //     grid_A.iam, grid_B.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    int nnzs[3]; nnzs[0] = nnz_A; nnzs[1] = nnz_B; nnzs[2] = nnz_C;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       NOW WE SOLVE THE ADI.
       ------------------------------------------------------------*/
    double* TTcores[3];
    int grid_main;
    fadi_ttsvd_3d_2grids_rep(options, m_A, nnz_A, nzval_A, rowind_A, colptr_A, m_B, nnz_B, nzval_B, rowind_B, colptr_B,
        m_C, nnz_C, nzval_C, rowind_C, colptr_C, &grid1, &grid2, U1, ldu1, V1, ldv1, U2, ldu2, V2, ldv2,
        pp1, qq1, ll1, pp2, qq2, ll2, tol, la[0], ua[0], lb[0], ub[0], la[1], ua[1], &(TTcores[0]), &(TTcores[1]), &(TTcores[2]), 
        r1, r2, &rank1, &rank2, grid_proc, rep, &grid_main);

    if (!iam1) {
        printf("Get solution!\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // printf("id in grid_A is %d, id in grid_B is %d, id in grid_C is %d, finishes solving.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       CHECK ACCURACY OF REPRODUCED RHS.
       ------------------------------------------------------------*/
    
    int ranks[2]; ranks[0] = rank1; ranks[1] = rank2;
    int locals[3]; locals[0] = ldu1; locals[2] = ldv2;
    if (grid_main == 0) {
        locals[1] = ldu2 / m_A;
    }
    else if (grid_main == 1) {
        locals[1] = ldv1 / m_C;
    }
    dcheck_error_TT_2grids_comb(ms, nnzs, nzvals, rowinds, colptrs, &grid1, &grid2, ranks, locals, 3, 
        trueF_global, TTcores, trueX_global, grid_proc, grid_main);
    
    // printf("id in grid_A is %d, id in grid_B is %d, id in grid_C is %d, finishes checking errors.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    if (iam1 != -1) {
        SUPERLU_FREE(U1);
        SUPERLU_FREE(U1_global);
        SUPERLU_FREE(pp1);
        SUPERLU_FREE(qq1);
        SUPERLU_FREE(TTcores[0]);
        SUPERLU_FREE(U2);
        SUPERLU_FREE(U2_global);
        SUPERLU_FREE(pp2);
        SUPERLU_FREE(qq2);
        if (grid_main == 0) {
            SUPERLU_FREE(TTcores[1]);
        }
    }
    if (iam2 != -1) {
        SUPERLU_FREE(V1);
        SUPERLU_FREE(V1_global);
        SUPERLU_FREE(V2);
        SUPERLU_FREE(V2_global);
        SUPERLU_FREE(pp1);
        SUPERLU_FREE(qq1);
        SUPERLU_FREE(pp2);
        SUPERLU_FREE(qq2);
        if (grid_main == 1) {
            SUPERLU_FREE(TTcores[1]);
        }
        SUPERLU_FREE(TTcores[2]);
    }
    
    if (!iam1) {
        SUPERLU_FREE(trueF_global);
        SUPERLU_FREE(trueX_global);
    }
    if (!iam2) {
        SUPERLU_FREE(trueF_global);
        SUPERLU_FREE(trueX_global);
    }

    fclose(fp_A);
    fclose(fp_B);
    fclose(fp_C);
    fclose(fp_shift1);
    fclose(fp_shift2);
    fclose(fp_int);
    fclose(fp_U1);
    fclose(fp_U2);
    fclose(fp_V1);
    fclose(fp_V2);
    fclose(fp_F);
    fclose(fp_X);
    fclose(fp_M);
    fclose(fp_R);

    // printf("id in grid_A is %d, id in grid_B is %d, id in grid_C is %d, releases memory.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    adi_gridexit_matrix(&grid1, &grid2);
    SUPERLU_FREE(grids);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

}
