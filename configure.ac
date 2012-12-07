## This scipt is a modification of the configure.in file
## from the magma package (version 0.2.1)
## http://cran.r-project.org/web/packages/magma/
## The copyright information for the magma package is as follows:
## Copyright (C) 2010 Brian J Smith
## Licensed under GNU GPL 3 or later

AC_INIT([permGPU], 0.14.3)
#AC_CONFIG_AUX_DIR([tools])

#################### GCC Compiler ####################

AC_PROG_CC(gcc)


#################### CUDA Toolkit ####################

AC_ARG_WITH([cuda-home],
	[AC_HELP_STRING([--with-cuda-home=DIR],
		[full path to CUDA installation @<:@/usr/local/cuda@:>@])],
	[CUDA_HOME="$withval"], [CUDA_HOME=/usr/local/cuda])
AC_MSG_RESULT([setting CUDA home directory to ${CUDA_HOME}])

NVCC=${CUDA_HOME}/bin/nvcc
AC_CHECK_FILE([${NVCC}],,
	[AC_MSG_ERROR([nvcc not found!])])

NCFLAGS="-arch=sm_20 -O2"
CUDA_INC=${CUDA_HOME}/include
AC_MSG_CHECKING([for CUDA headers])
AC_CHECK_FILE([${CUDA_INC}/cublas.h],
	[NCFLAGS="${NCFLAGS} -I${CUDA_INC}"],
	[AC_MSG_ERROR([CUDA headers not found!])]
)

AC_MSG_CHECKING([whether this is a 64-bit version of CUDA])
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [check_bit=64-bit])
if test "${check_bit}" == "64-bit"; then
	LDFLAGS="-L${CUDA_HOME}/lib64"
else
	LDFLAGS="-L${CUDA_HOME}/lib"
fi

LIBS="-lcuda -lcublas"

: ${R_HOME=`R RHOME`}
AC_ARG_WITH([r-home],
	[AC_HELP_STRING([--with-r-home=DIR],
		[full path to R installation @<:@@:>@])],
	[R_HOME=$withval]
)
AC_MSG_RESULT([setting R home directory to ${R_HOME}])
AC_CHECK_PROG([check_R], [R], [yes],, [${R_HOME}/bin])
if test -z "${check_R}"; then
	AC_MSG_ERROR([R not found!])
fi 

AC_MSG_RESULT([setting R header files to ${R_INCLUDE_DIR}])
R_CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
NCFLAGS="${NCFLAGS} -Xcompiler \"-O2 ${R_CPICFLAGS} -I${R_INCLUDE_DIR}\""


AC_SUBST([NVCC])
AC_SUBST([NCFLAGS])
AC_SUBST([LDFLAGS])
AC_SUBST([LIBS])

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT