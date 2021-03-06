\name{permGPU-package}
\alias{permGPU-package}
\alias{permGPU}
\docType{package}
\title{
packageTitle{permGPU}
}
\description{
  This package can be used to carry out permutation
  resampling inference using GPUs, as described in 
  I. D. Shterev, S. H. Jung, S. L. George and K. Owzar. 
  "permGPU: Using Graphics Processing Units in RNA Microarray Association Studies",
  BMC Bioinformatics 11(1), 2010. Currently the package supports six
  test statistics: the t and Wilcoxon tests, for two-sample problems,
  the Pearson and Spearman statistics, for non-censored continuous
  outcomes, and the Cox score and rank score test (Jung et al, 2005), for
  right-censored time-to-event outcomes. In addition to the
  test statistics and the corresponding marginal permutation
  P-values, the package produces family-wise error adjusted
  P-values using a single-step procedure (Westfall and Young, 1993).
}
\details{

The DESCRIPTION file:
packageDESCRIPTION{permGPU}
packageIndices{permGPU}

}

\note{
To build this package, the CUDA SDK (version 2.3 or higher) must be installed
on the system. Specifically, the nvcc compiler must be in the path and
the CUDA_HOME must be properly defined. For example, if the SDK kit is
installed under /usr/local/cuda then the CUDA_HOME variable needs to be
set to  /usr/local/cuda . The SDK can be obtained from \url{https://www.nvidia.com}
The CUDA_HOME variable can also be explicitly defined in \code{permGPU/src/Makefile}.
The maximum number of patients for the current version is \eqn{1000}.

The \code{R} environment variables \code{R_LIB} and \code{R_INCLUDE}
need to be correctly configured to build the package from source.
Alternatively, these can be set in \code{permGPU/src/Makefile}.


To build this package, a number of C++ classes and functions for random
number generation (available from \url{https://www.agner.org/random/}
under a GPL license) and a C++ template for calculating ranks
(available from
\url{https://sites.google.com/site/jivsoft/Home/compute-ranks-of-elements-in-a-c---array-or-vector}
under a BSD license) are needed. The requisite files are included in the
package source code tar ball. In future releases, these functionalities
will be replaced by native \code{R} functions from \code{R.h}
and \code{Rmath.h}. 
}

\author{
  I. D. Shterev, S.-H. Jung, S. L. George and K. Owzar
  
  Maintainer: I. D. Shterev <i.shterev@duke.edu>
}



\references{
   Shterev, I.D., Jung, S.-H., George S.L., Owzar K.
   permGPU: Using graphics processing units in RNA microarray 
   association studies. BMC Bioinformatics 2010, 11:329.

  Jung, S.-H., Owzar K., George, S.L. (2005). A multiple testing
  procedure to associate gene expression levels with 
  survival. Statistics in Medicine. 24(20), 3077--88.

  Westfall, P.H. and Young, S.S. (1993).
  Resampling-Based Multiple Testing: Examples and Methods for
  P-value Adjustment, Wiley-Interscience, New York.  
}

\keyword{ package }

