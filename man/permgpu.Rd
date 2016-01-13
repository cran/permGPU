\name{permgpu}
\alias{permgpu}
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
permgpu(datobj, y, event = NULL, test, B, diag = FALSE, scale = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datobj}{
    %%     ~~Describe \code{datobj} here~~
    an \code{\link[Biobase:class.ExpressionSet]{ExpressionSet}} object
    containing the expression and phenotype data.
}
  \item{y}{
    %%     ~~Describe \code{y} here~~
    The name of the outcome variable. Note that y must be an
    element of names(pData(datobj)). See example.
}
  \item{event}{
    %%     ~~Describe \code{event} here~~
    In the case of survival analysis, event is the name of
    the event indicator. Note that event must be an
    element of names(pData(datobj)). See example.
}
  \item{test}{
    %%     ~~Describe \code{test} here~~
    specifies the test to be performed. Possible options are
    \code{ttest} (two-sample t-test),
    \code{wilcoxon} (two-sample Wilcoxon test),
    \code{pearson} (Pearson correlation test),
    \code{spearman} (Spearman rank correlation test),
    \code{cox} (Cox score test)
    and \code{npcox} (Cox rank score test). 
}
  \item{B}{
    %%     ~~Describe \code{B} here~~
    specifies the number of random permutations to be performed.
}
  \item{diag}{
    %%     ~~Describe \code{diag} here~~
    This flag can be set to TRUE if 
    specifies the type of object returned.
  }
 \item{scale}{
    %%     ~~Describe \code{test} here~~
    If TRUE, markers are centered. 
}
}

\value{
  %%  ~Describe the value returned
  This function returns a data frame. The first column contains the gene
  names. The second, thrid and fourth columns contain the marginal test
  statistics, marginal  unadjusted permutation P-values and FWER
  adjusted P-values respectively.
  If diag=TRUE, this function returns a list consisting of the
  following elements:
  \item{RESULTS}{The results data frame as described above}
  \item{EXPR}{The gene expression data}
  \item{y}{The outcome data}
  \item{event}{event indicator(s) for survival analysis}
  \item{n}{The number of patients}
  \item{K}{The number of genes}
  \item{B}{The number of permutations}
  \item{test}{The test used in the permutation analysis}
%% ...
}
\note{
  The maximum number of patients for the current version is \eqn{1000}.
}
\references{
  %% ~put references to the literature/web site here ~
   Jung, S.-H., Owzar K., George, S.L. (2005)
   A multiple testing procedure to associate gene expression levels with 
   survival.\emph{Statistics in Medicine}. \bold{24}, 20, 3077--88.

   Shterev, I.D., Jung, S.-H., George S.L., Owzar K.
   permGPU: Using graphics processing units in RNA microarray 
   association studies. BMC Bioinformatics 2010, 11:329.

   For the Director's Challenge Consortium 
   for the Molecular Classification of Lung Adenocarcinoma, Shedden K., Taylor
   J.M.G., Enkemann S.A., Tsao M.S., Yeatman T.J., Gerald W.L., Eschrich S., Jurisica I.,
   Giordano T.J., Misek D.E., Chang A.C., Zhu C.Q., Strumpf D., Hanash S., Shepherd F.A.,
   Ding K., Seymour L., Naoki K., Pennell N., Weir B., Verhaak R., Ladd-Acosta C.,
   Golub T., Gruidl M., Sharma A., Szoke J., Zakowski M., Rusch V., Kris M., Viale A.,
   Motoi N., Travis W., Conley B., Seshan V.E., Meyerson M., Kuick R., Dobbin K.K.,
   Lively T., Jacobson J.W., Beer D.G. (2008) 
   Gene expression-based survival
   prediction in lung adenocarcinoma: a multi-site, blinded validation study.
   \emph{Nat Med}. \bold{14}, 8, 822--827.


}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\examples{
library(Biobase)
set.seed(123)

## Generate toy phenotype and expression data sets
## This example consists of 4 markers and 100 patients
## grp is a binary trait (e.g., case vs control)
## bp is a continuous trait (e.g., blood pressure)
## ostime is a right-censored time-to-event trait (e.g., observed
## time of death)
## event is the event indicator (1=dead or 0=censored) for ostime

n<-100
K<-4
grp=rep(1:0,each=n/2)
bp=rnorm(n)
atime=rexp(n)
ctime=runif(n,0,1)
otime=pmin(atime,ctime)
event=as.integer(atime<=ctime)
pdat=data.frame(grp,bp,otime,event)
rm(grp,atime,ctime,otime,event)
expdat=matrix(rnorm(K*n),K,n)

## Assign marker names g1,...,gK to the expression data set and
## patient ids id1,...,idn to the expression and phenotype data
rownames(expdat)=paste("g",1:K,sep="")
patid=paste("id",1:n,sep="")
rownames(pdat)=patid
colnames(expdat)=patid

## Create the ExprSet object
testdat=makeExprSet(expdat,pdat)
class(testdat)

## Carry out permutation analysis with grp as the outcome
## using the two-sample t-test with B=100 random permutations
permgpu(testdat,"grp",B=100,test="ttest")

## Carry out permutation analysis with grp as the outcome
## using the two-sample Wilcoxon with B=100 random permutations
permgpu(testdat,"grp",B=100,test="wilcoxon")

## Carry out permutation analysis with bp as the outcome
## using the Pearson test with B=100 random permutations
permgpu(testdat,"bp",B=100,test="pearson")

## Carry out permutation analysis with bp as the outcome
## using the Spearman test with B=100 random permutations
permgpu(testdat,"bp",B=100,test="spearman")

## Carry out permutation analysis with ostime as the outcome
## using the covariance test (Jung et al, 2005) with B=100 
## random permutations.
permgpu(testdat,"otime",event="event",B=100,test="cox")

## Carry out permutation analysis with ostime as the outcome
## using the rank-covariance test (Jung et al, 2005) with B=100 
## random permutations.
permgpu(testdat,"otime",event="event",B=100,test="npcox")

## To carry out the analyses for the Director's Challenge
## Consortium Lung Cancer data, download the RMA pre-processed
## expressionSet object from the project webpage
## http://code.google.com/p/permgpu/
## After attaching it, check the md5sum signature
## attach("RMADAT-DCHALL.RData")
## md5sum("RMADAT-DCHALL.RData")
## 404fc27fe0c6d11c844e06139912f7ca
## A Sweave file outlining the steps carried out to pre-process
## the data is available from the project page.
##
## To carry out association testing using the Cox score test
## permgpu(RMADAT,"ostime",event="event",B=10000,test="cox")
## To carry out association testing using the Cox rank score test
## permgpu(RMADAT,"ostime",event="event",B=10000,test="npcox")




}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{survival}
\keyword{nonparametric}% __ONLY ONE__ keyword per line
