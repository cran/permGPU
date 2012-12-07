test.scoregpu = function(){
  outcome = rep(1,20)/10
  event = rep(c(0,1), 10)
  expr = matrix(seq(1,40), 2, 20)
  checkEquals(sum(scoregpu(outcome, event, expr, test="npcox", B = 0, index = FALSE)), 1.567855, tolerance=0.001)
}
