.SUFFIXES: .f90 .o
FC = mpiifort
FFLAGS = -r8 -O2 -fPIC -fp-model=precise
FFLAGS += -warn all -WB
ARCH = 
EXES = main.e
OBJS = tdma_parallel.o main.o 
SRCS = $(OBJS:.o=.f90)

all: $(EXES)

$(EXES): $(OBJS)
	$(FC) -o $(EXES) $(OBJS)

.f90.o:
	$(FC) $(FFLAGS) -c $<

main.o: main.f90
tdma_parallel.o: tdma_parallel.f90
%.o: %.mod

clean:
	rm -f *.o *.mod $(EXES)
