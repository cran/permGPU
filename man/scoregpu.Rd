\name{scoregpu}
\alias{scoregpu}

\title{
  Computes score test statistic using permGPU
}
\description{
  This function can be used to carry out score test inference with GPUs. Currently the function supports two test
  statistics: the Cox score and rank score tests (Jung et al, 2005), for
  right-censored time-to-event outcomes.
}
\usage{
scoregpu(y, event, markers, test, B=0, stand=TRUE, pval=TRUE, index=FALSE, scale=FALSE)
}

\arguments{
  \item{y}{
    The name of the outcome variable.
}
  \item{event}{
    In the case of survival analysis, event is the name of
    the event indicator.
}
 \item{markers}{
    Expression matrix. Probes are along rows.
}
  \item{test}{
    specifies the test to be performed. Possible options are
    \code{cox} (Cox score test)
    and \code{npcox} (Cox rank score test). 
}
 \item{B}{
    number of permutations. 
}
 \item{stand}{
    If TRUE, the squared test statistic is returned.
}
 \item{pval}{
    If TRUE, the p-value is returned (\code{stand} has to be FALSE). 
}
 \item{index}{
    If TRUE, the sample indexes at each permutation are returned. 
}
 \item{scale}{
    If TRUE, markers are centered. 
}
}

\value{
  This function returns a data frame with one column of test statistics.
}
\note{
  The maximum number of patients for the current version is \eqn{1000}.
}
\references{
   Jung, S.-H., Owzar K., George, S.L. (2005)
   A multiple testing procedure to associate gene expression levels with 
   survival.\emph{Statistics in Medicine}. \bold{24}, 20, 3077--88.

   Shterev, I.D., Jung, S.-H., George S.L., Owzar K.
   permGPU: Using graphics processing units in RNA microarray 
   association studies. BMC Bioinformatics 2010, 11:329.
}

\examples{
set.seed(123)
n<-100
K<-3

x1<-matrix(rnorm(n*K),K,n)
x2<-matrix(rnorm(n*K),K,n)
x3<-matrix(rnorm(n*K),K,n)
otime<-rexp(n)
event<-rbinom(n,1,0.8)

rownames(x1)<-paste("g",1:K,sep="")
colnames(x1)<-paste("p",1:n,sep="")
rownames(x2)<-paste("g",1:K,sep="")
colnames(x2)<-paste("p",1:n,sep="")
rownames(x3)<-paste("g",1:K,sep="")
colnames(x3)<-paste("p",1:n,sep="")

x=list(x1,x2,x3)
library(foreach)

# carry out analysis with npcox test
foreach(i=1:length(x)) %do% scoregpu(otime,event,x[[i]],"npcox")

# carry out analysis with npcox test using B=10 permutations
foreach(i=1:length(x)) %do% scoregpu(otime,event,x[[i]],"npcox", B=10)
}
