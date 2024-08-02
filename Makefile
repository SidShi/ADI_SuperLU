IDIR=/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/include
LDIR=/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/lib
LIBSUPERLU=-lsuperlu_dist
ISUPERLU=-I$(IDIR)
LSUPERLU=-L$(LDIR)
DRIVERS=adi_mat fadi_mat adi_mat_shifts fadi_ttsvd_3d fadi_ttsvd_3d_2grids fadi_ttsvd_3d_rep fadi_para_ttsvd_3d
DRIVERS+=fadi_ttsvd_4d fadi_ttsvd_4d_rep fadi_ttsvd_4d_2way
DRIVERS+=fadi_ttsvd_5d_rep fadi_ttsvd_5d_2way

CC=cc

.PHONY: all
all: adi_mat fadi_mat adi_mat_shifts fadi_ttsvd_3d fadi_ttsvd_3d_2grids fadi_ttsvd_3d_rep fadi_para_ttsvd_3d\ 
fadi_ttsvd_4d fadi_ttsvd_4d_rep fadi_ttsvd_4d_2way fadi_ttsvd_5d_rep fadi_ttsvd_5d_2way

adi_mat: driver_adi.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

adi_mat_shifts: driver_adi_shifts.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_mat: driver_fadi.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_3d: driver_ttsvd_fadi.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_3d_2grids: driver_ttsvd_fadi_2grids.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_3d_rep: driver_ttsvd_fadi_rep.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_para_ttsvd_3d: driver_para_ttsvd_fadi.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_4d: driver_ttsvd_fadi_4D.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_4d_rep: driver_ttsvd_fadi_4D_rep.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_4d_2way: driver_ttsvd_fadi_4D_2way.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_5d_rep: driver_ttsvd_fadi_5D_rep.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

fadi_ttsvd_5d_2way: driver_ttsvd_fadi_5D_2way.o adi.o adi_grid.o adi_helper.o read_equation.o
	$(CC) -o $@ $^ $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

adi.o: adi.c adi.h adi_helper.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

adi_grid.o: adi_grid.c adi_grid.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

adi_helper.o: adi_helper.c adi_helper.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

read_equation.o: read_equation.c read_equation.h adi_helper.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_adi.o: driver.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_adi_shifts.o: driver_shifts.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_fadi.o: driver_fadi.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi.o: driver_ttsvd_fadi.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_2grids.o: driver_ttsvd_fadi_2grids.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_rep.o: driver_ttsvd_fadi_rep.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_para_ttsvd_fadi.o: driver_ttsvd_para_fadi.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_4D.o: driver_ttsvd_fadi_4D.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_4D_rep.o: driver_ttsvd_fadi_4D_rep.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_4D_2way.o: driver_ttsvd_fadi_4D_2way.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_5D_rep.o: driver_ttsvd_fadi_5D_rep.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

driver_ttsvd_fadi_5D_2way.o: driver_ttsvd_fadi_5D_2way.c adi.h adi_grid.h read_equation.h
	$(CC) -c -o $@ $< $(ISUPERLU) $(LSUPERLU) $(LIBSUPERLU)

.PHONY: clean
clean:
	rm -f *.o $(DRIVERS)