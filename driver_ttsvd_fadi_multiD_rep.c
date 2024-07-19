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
    int    m_A, m_B, m_C, m_D, mm, rr;
    int    dim = 4;
    int    ms[dim], ms_rev[dim], rs[dim-1], locals1[dim], locals2[dim], locals2_alt[dim], nrhss[dim-1], nnzs[dim];
    int    nprow1, npcol1, nprow2, npcol2, lookahead, colperm, rowperm, ir, symbfact, rep;
    int    nprows[2], npcols[2], grid_proc[2];
    int    iam1, iam2, info, grid_main;
    double *nzvals[dim], *nzvals_alt[dim];
    int_t  *rowinds[dim], *colptrs[dim], *rowinds_alt[dim], *colptrs_alt[dim];
    char   **cpp, c, *suffix;
    char   postfix[10];
    FILE   *fp_A, *fp_B, *fp_C, *fp_D, *fp_shift1, *fp_shift2, *fp_shift3, *fp_int;
    FILE   *fp_U1, *fp_U2, *fp_U3, *fp_V1, *fp_V2, *fp_V3, *fp_X, *fp_F, *fp_M, *fp_R, *fopen();
    int ii, omp_mpi_level;
    int p; /* The following variables are used for batch solves */
    int_t i, j;
    double *pps[dim-1], *qqs[dim-1];
    double *TTcores[dim];
    double *trueX_global, *U1_global, *U2_global, *U3_global, *V1_global, *V2_global, *V3_global, *trueF_global;
    double *Us[dim-1], *Vs[dim-1];
    double tol = 1.0/(1e16);
    double las[2*(dim-2)], uas[2*(dim-2)], lbs[2*(dim-2)], ubs[2*(dim-2)];
    int_t lls[dim-1];
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

        char buf_V1[3];
        buf_V1[0] = 'V'; buf_V1[1] = 'a'; buf_V1[2] = '\0';
        char name_V1[100];
        strcpy(name_V1, *cpp);
        strcat(name_V1, buf_V1);
        strcat(name_V1, postfix);
        if (!(fp_V1 = fopen (name_V1, "r")))
        {
            ABORT ("File for RHS second factor of first unfolding does not exist");
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

        char buf_V3[3];
        buf_V3[0] = 'V'; buf_V3[1] = 'c'; buf_V3[2] = '\0';
        char name_V3[100];
        strcpy(name_V3, *cpp);
        strcat(name_V3, buf_V3);
        strcat(name_V3, postfix);
        if (!(fp_V3 = fopen (name_V3, "r")))
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

    dread_size(fp_M, dim, ms, grids);
    dread_size(fp_R, dim-1, nrhss, grids);
    m_A = ms[0]; m_B = ms[1]; m_C = ms[2]; m_D = ms[3];
    ms_rev[0] = ms[3]; ms_rev[1] = ms[2]; ms_rev[2] = ms[1]; ms_rev[3] = ms[0];

    // printf("Process with id %d in grid1, %d in grid2 gets problem size %d, %d, %d, %d, and rhs rank %d, %d, %d.\n", 
    //     grid1.iam, grid2.iam, ms[0], ms[1], ms[2], ms[3], nrhss[0], nrhss[1], nrhss[2]);
    // fflush(stdout);

    if ((iam1 != -1) || (iam2 != -1)) {
        dread_shift_twogrids(fp_shift1, &(pps[0]), &(qqs[0]), &(lls[0]), &grid1, &grid2, 0, 1, grid_proc);
        dread_shift_twogrids(fp_shift2, &(pps[1]), &(qqs[1]), &(lls[1]), &grid1, &grid2, 0, 1, grid_proc);
        dread_shift_twogrids(fp_shift3, &(pps[2]), &(qqs[2]), &(lls[2]), &grid1, &grid2, 0, 1, grid_proc);

        // printf("Proc %d in grid1 and %d in grid2 gets first elements of third shift with length %d to be %f and %f.\n", 
        //     iam1, iam2, lls[2], pps[2][0], qqs[2][0]);
        // fflush(stdout);

        if (!iam1) {
            printf("Read shifts for all three equations!\n");
            fflush(stdout);
        }
    
        dread_shift_multi_interval_2grids_2way(fp_int, dim, las, uas, lbs, ubs, &grid1, &grid2, grid_proc);

        // printf("Proc %d in grid1 and %d in grid2 gets final spectra intervals to use to be %f, %f, %f, and %f.\n", 
        //     iam1, iam2, las[1], uas[1], lbs[1], ubs[1]);
        // fflush(stdout);

        if (!iam1) {
            // printf("Read spectra bound of A and B to be %f, %f, %f, %f!\n", la, ua, lb, ub);
            printf("Read spectra bound of combinations of matrices!\n");
            fflush(stdout);
        }
    }
    
    if (iam1 != -1) {
        dread_RHS_factor(fp_U1, &(Us[0]), &grid1, &mm, &rr, &(locals1[0]), &U1_global);
        locals2[3] = locals1[0];
        // printf("Grid_A proc %d gets first and last element of U to be %f and %f.\n", iam_A, U[0], U[ldu*r-1]);
        // fflush(stdout);
    
        dread_RHS_factor_multidim(fp_U2, &(Us[1]), &grid1, ms, 1, &(locals1[1]), &U2_global);
        // printf("Grid_B proc %d gets first and last element of V to be %f and %f.\n", iam_B, V[0], V[ldv*r-1]);
        // fflush(stdout);

        dread_RHS_factor_multidim(fp_U3, &(Us[2]), &grid1, ms, 2, &(locals1[2]), &U3_global);

        if (!iam1) {
            printf("Read U1, U2, and U3!\n");
            fflush(stdout);
        }
    }

    if (iam2 != -1) {
        dread_RHS_factor(fp_V1, &(Vs[0]), &grid2, &mm, &rr, &(locals2[0]), &V1_global);
        locals1[3] = locals2[0];
        // printf("Grid_A proc %d gets first and last element of U to be %f and %f.\n", iam_A, U[0], U[ldu*r-1]);
        // fflush(stdout);
    
        dread_RHS_factor_multidim(fp_V2, &(Vs[1]), &grid2, ms_rev, 1, &(locals2[1]), &V2_global);
        // printf("Grid_B proc %d gets first and last element of V to be %f and %f.\n", iam_B, V[0], V[ldv*r-1]);
        // fflush(stdout);

        dread_RHS_factor_multidim(fp_V3, &(Vs[2]), &grid2, ms_rev, 2, &(locals2[2]), &V3_global);

        if (!iam2) {
            printf("Read V1, V2, and V3!\n");
            fflush(stdout);
        }
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
    dread_matrix(fp_A, suffix, &grid1, &mm, &rr, &(nnzs[0]), &(nzvals[0]), &(rowinds[0]), &(colptrs[0]));
    if (!iam1) {
        printf("Read A!\n");
        fflush(stdout);
    }
    if (iam1 != -1) {
        dallocateA_dist(m_A, nnzs[0], &(nzvals_alt[0]), &(rowinds_alt[0]), &(colptrs_alt[0]));
        for (i = 0; i < m_A; ++i) {
            for (j = colptrs[0][i]; j < colptrs[0][i+1]; ++j) {
                nzvals_alt[0][j] = nzvals[0][j];
                rowinds_alt[0][j] = rowinds[0][j];
            }
            colptrs_alt[0][i] = colptrs[0][i];
        }
        colptrs_alt[0][m_A] = colptrs[0][m_A];
    }

    dread_matrix_twogrids(fp_B, suffix, &grid1, &grid2, &mm, &rr, &(nnzs[1]), &(nzvals[1]), &(rowinds[1]), &(colptrs[1]));
    dread_matrix_twogrids(fp_C, suffix, &grid1, &grid2, &mm, &rr, &(nnzs[2]), &(nzvals[2]), &(rowinds[2]), &(colptrs[2]));
    if (!iam1) {
        printf("Read B and C!\n");
        fflush(stdout);
    }
    if ((iam1 != -1) || (iam2 != -1)) {
        dallocateA_dist(m_B, nnzs[1], &(nzvals_alt[1]), &(rowinds_alt[1]), &(colptrs_alt[1]));
        for (i = 0; i < m_B; ++i) {
            for (j = colptrs[1][i]; j < colptrs[1][i+1]; ++j) {
                nzvals_alt[1][j] = nzvals[1][j];
                rowinds_alt[1][j] = rowinds[1][j];
            }
            colptrs_alt[1][i] = colptrs[1][i];
        }
        colptrs_alt[1][m_B] = colptrs[1][m_B];

        dallocateA_dist(m_C, nnzs[2], &(nzvals_alt[2]), &(rowinds_alt[2]), &(colptrs_alt[2]));
        for (i = 0; i < m_C; ++i) {
            for (j = colptrs[2][i]; j < colptrs[2][i+1]; ++j) {
                nzvals_alt[2][j] = nzvals[2][j];
                rowinds_alt[2][j] = rowinds[2][j];
            }
            colptrs_alt[2][i] = colptrs[2][i];
        }
        colptrs_alt[2][m_C] = colptrs[2][m_C];
    }

    dread_matrix(fp_D, suffix, &grid2, &mm, &rr, &(nnzs[3]), &(nzvals[3]), &(rowinds[3]), &(colptrs[3]));
    if (!iam2) {
        printf("Read D!\n");
        fflush(stdout);
    }
    if (iam2 != -1) {
        dallocateA_dist(m_D, nnzs[3], &(nzvals_alt[3]), &(rowinds_alt[3]), &(colptrs_alt[3]));
        for (i = 0; i < m_D; ++i) {
            for (j = colptrs[3][i]; j < colptrs[3][i+1]; ++j) {
                nzvals_alt[3][j] = nzvals[3][j];
                rowinds_alt[3][j] = rowinds[3][j];
            }
            colptrs_alt[3][i] = colptrs[3][i];
        }
        colptrs_alt[3][m_D] = colptrs[3][m_D];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       NOW WE SOLVE THE ADI.
       ------------------------------------------------------------*/
    fadi_ttsvd_rep(options, dim, ms, nnzs, nzvals, rowinds, colptrs, &grid1, &grid2, Us, Vs, locals1, locals2, nrhss, pps, qqs, lls, tol,
        las, uas, lbs, ubs, TTcores, rs, grid_proc, rep, &grid_main);
    
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
    
    if (grid_main == 0) {
        dcheck_error_TT_2grids_comb(ms, nnzs, nzvals_alt, rowinds_alt, colptrs_alt, &grid1, &grid2, rs, locals1, dim, 
            trueF_global, TTcores, trueX_global, grid_proc, grid_main);
    }
    else if (grid_main == 1) {
        for (j = 0; j < dim; ++j) {
            locals2_alt[j] = locals2[dim-1-j];
        }
        dcheck_error_TT_2grids_comb(ms, nnzs, nzvals_alt, rowinds_alt, colptrs_alt, &grid1, &grid2, rs, locals2_alt, dim, 
            trueF_global, TTcores, trueX_global, grid_proc, grid_main);
    }
    // printf("id in grid_A is %d, id in grid_B is %d, id in grid_C is %d, finishes checking errors.\n", 
    //     grid_A.iam, grid_B.iam, grid_C.iam);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    if (iam1 != -1) {
        SUPERLU_FREE(Us[0]);
        SUPERLU_FREE(U1_global);
        SUPERLU_FREE(pps[0]);
        SUPERLU_FREE(qqs[0]);
        SUPERLU_FREE(TTcores[0]);
        SUPERLU_FREE(Us[1]);
        SUPERLU_FREE(U2_global);
        SUPERLU_FREE(pps[1]);
        SUPERLU_FREE(qqs[1]);
        SUPERLU_FREE(Us[2]);
        SUPERLU_FREE(U3_global);
        SUPERLU_FREE(pps[2]);
        SUPERLU_FREE(qqs[2]);
        if (grid_main == 0) {
            SUPERLU_FREE(TTcores[1]);
            SUPERLU_FREE(TTcores[2]);
        }
    }
    if (iam2 != -1) {
        SUPERLU_FREE(Vs[0]);
        SUPERLU_FREE(V1_global);
        SUPERLU_FREE(pps[0]);
        SUPERLU_FREE(qqs[0]);
        SUPERLU_FREE(TTcores[3]);
        SUPERLU_FREE(Vs[1]);
        SUPERLU_FREE(V2_global);
        SUPERLU_FREE(pps[1]);
        SUPERLU_FREE(qqs[1]);
        SUPERLU_FREE(Vs[2]);
        SUPERLU_FREE(V3_global);
        SUPERLU_FREE(pps[2]);
        SUPERLU_FREE(qqs[2]);
        if (grid_main == 1) {
            SUPERLU_FREE(TTcores[1]);
            SUPERLU_FREE(TTcores[2]);
        }
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
    fclose(fp_D);
    fclose(fp_shift1);
    fclose(fp_shift2);
    fclose(fp_shift3);
    fclose(fp_int);
    fclose(fp_U1);
    fclose(fp_U2);
    fclose(fp_U3);
    fclose(fp_V1);
    fclose(fp_V2);
    fclose(fp_V3);
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
