#define CLASS 'A'
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   */
   
#define COMPILETIME "13 May 2023"
#define NPBVERSION "3.4.2"
#define CC "gcc"
#define CFLAGS "-O3 -fopenmp -mcmodel=large"
#define CLINK "$(CC)"
#define CLINKFLAGS "$(CFLAGS)"
#define C_LIB "-lm -L$(PAPI_LIB) -lpapi"
#define C_INC "-I$(PAPI_INC)"
