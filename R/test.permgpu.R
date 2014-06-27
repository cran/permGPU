test.permgpu = function(test){
  otime = rep(1,20)/seq(1,20)
  event = rep(c(0,1), 10)
  pdat = data.frame(otime, event)
  expr = matrix(seq(1,40)/seq(40,1), 2, 20)

  rownames(expr) = paste("g", 1:2, sep="")
  patid = paste("id", 1:20, sep="")
  rownames(pdat) = patid
  colnames(expr) = patid

  testdat = makeExprSet(expr, pdat)

  if (test == "ttest")
    checkEquals(sum(permgpu(testdat, "event", B=0, test="ttest")[["stat"]]), -1.439, tolerance=0.001)
  else if (test == "wilcoxon"){
    n0 = sum(event==0)
    n1 = sum(event==1)
    m  = n1 * (n0+n1+1.0) / 2.0
    std = sqrt(n0*n1*(n0+n1+1.0)/12.0)
    checkEquals(sum(permgpu(testdat, "event", B=0, test="wilcoxon")[["stat"]]), sum((sum(rank(exprs(testdat)[1,])[event==1])-m)/std, (sum(rank(exprs(testdat)[2,])[event==1])-m)/std), tolerance=0.001)
  }
  else if (test == "npcox"){
    checkEquals(sum(permgpu(testdat, "otime", "event", B=0, test="npcox")[["stat"]]), 39.21569, tolerance=0.001)
  }
  else if (test == "cox"){
    checkEquals(sum(permgpu(testdat, "otime", "event", B=0, test="cox")[["stat"]]), 57.21867, tolerance=0.001)
  }
  else
    stop("test unknown.")
}
