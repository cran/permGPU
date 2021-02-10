\name{test.scoregpu}
\alias{test.scoregpu}

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
test.scoregpu()
}

\examples{
  test.scoregpu()
}

