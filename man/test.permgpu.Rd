\name{test.permgpu}
\alias{test.permgpu}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Conduct permutation resampling analysis using permGPU
}
\description{
  This function can be used to carry our permutation
  resampling inference with GPUs. Currently the function supports six test
  statistics: the t and Wilcoxon tests, for two-sample problems,
  the Pearson and Spearman statistics, for non-censored continuous
  outcomes, and the Cox score and rank score tests (Jung et al, 2005), for
  right-censored time-to-event outcomes.
}
\usage{
test.permgpu(test)
}
\arguments{
  \item{test}{
    %%     ~~Describe \code{y} here~~
    Specifies the test.
  }
}

\examples{
  # check permgpu ("ttest")
  test.permgpu("ttest")

  # check permgpu ("wilcoxon")
  test.permgpu("wilcoxon")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{survival}
\keyword{nonparametric}% __ONLY ONE__ keyword per line
