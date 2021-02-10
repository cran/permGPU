\name{makeExprSet}
\alias{makeExprSet}

\title{
Create an expressionSet object for use with \link{permGPU}
}
\description{
  The \link{permGPU} function expects that the phenotypic and
  molecular (expression) data are provided as an
  \code{\link[Biobase:class.ExpressionSet]{ExpressionSet}}.
  This is a simple utility function that creates this object
  so that permGPU can be used along with data objects
  created for use with certain Bioconductor packages.
}
\usage{
makeExprSet(exprdat, phenodat, anno = "custom")
}

\arguments{
  \item{exprdat}{
    This should be an \eqn{K \times n}, where \eqn{K} denotes the number
    of markers/features and \eqn{n} denotes the number of patients,
    expression matrix. It is expected that the \eqn{K} marker names
    are assigned as row names of this matrix (i.e., could be extracted
    as rownames(exprdat).
}
\item{phenodat}{
  This is an \eqn{n \times p} data.frame, \eqn{n} denotes the number
  of patients and \eqn{p} denotes the number of clinical co-variables.
  It is assumed that the rows of this data.frame are matched
  up with the columns of exprdat
}
  \item{anno}{
  This slot can be used to assign a label to the data set.
}
}

\value{
  An object of class \code{\link[Biobase:class.ExpressionSet]{ExpressionSet}}.
}

\note{
  This function may be deprecated in future releases if similar
  functionality is found in the base Bioconductor extension packages.
}


\seealso{
\code{\link[Biobase:class.ExpressionSet]{ExpressionSet}}
}
\examples{
library(Biobase)
set.seed(123)

## Generate toy phenotype and expression data sets
## This example consists of 4 markers and ten patients
n<-10
K<-4
pdat=data.frame(grp=rep(1:0,each=n/2),bp=rnorm(n),ostime=rexp(n),event=rbinom(n,1,0.8))
expdat=matrix(rnorm(K*n),K,n)

## Assign marker names g1,...,gK to the expression data set
## (as row names) and patient ids id1,...,idn to the expression 
## data set (as column names) and phenotype data (as row names)
rownames(expdat)=paste("g",1:K,sep="")
patid=paste("id",1:n,sep="")
rownames(pdat)=patid
colnames(expdat)=patid

## Create the ExprSet object
testdat=makeExprSet(expdat,pdat)
class(testdat)


## Check the dimensions of the expression and phenotype data sets
dim(exprs(testdat))
dim(pData(testdat))

## Get sample and marker ids
sampleNames(testdat)
featureNames(testdat)
}

