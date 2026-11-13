# Reference values of the KrigInv SUR criterion (Chevalier et al., 2014) for a
# DiceKriging model with FIXED hyper-parameters, exported to CSV for comparison
# with the Python implementation (validation/validate_sur_vs_kriginv.py).
#   Rscript validation/kriginv_reference.R <out_dir>
suppressMessages({library(DiceKriging); library(KrigInv)})
args <- commandArgs(trailingOnly = TRUE); out <- ifelse(length(args) > 0, args[1], "kriginv_ref")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(1)
d <- 2; n <- 20; M <- 400; K <- 40
f <- function(x) sin(2 * x[1]) + 0.5 * x[2]^2 - 0.3
X <- matrix(runif(n * d, -2, 2), ncol = d); colnames(X) <- c("x1", "x2")
y <- apply(X, 1, f)
theta <- c(1.1, 0.9); sig2 <- 1.3; mu0 <- 0.2; nug <- 1e-8
model <- km(~1, design = data.frame(X), response = y, covtype = "matern5_2",
            coef.trend = mu0, coef.cov = theta, coef.var = sig2, nugget = nug)
U  <- matrix(runif(M * d, -2, 2), ncol = d); colnames(U)  <- c("x1", "x2")
Xc <- matrix(runif(K * d, -2, 2), ncol = d); colnames(Xc) <- c("x1", "x2")
pred <- KrigInv:::predict_nobias_km(object = model, newdata = U, type = "UK", se.compute = TRUE)
pn <- KrigInv:::excursion_probability(mn = pred$mean, sn = pred$sd, T = 0)
precalc <- KrigInv:::precomputeUpdateData(model, U)
current.sur <- mean(pn * (1 - pn))
crit <- numeric(K); sx2 <- numeric(K); KN <- matrix(0, M, K)
for (k in 1:K) {
  xk <- Xc[k, , drop = FALSE]
  kr <- KrigInv:::predict_nobias_km(object = model, newdata = as.data.frame(xk), type = "UK",
                                    se.compute = TRUE, cov.compute = TRUE)
  sx2[k] <- as.numeric(kr$cov)
  KN[, k] <- KrigInv:::computeQuickKrigcov(model, U, xk, precalc, kr$F.newdata, kr$c)
  crit[k] <- KrigInv:::sur_optim_parallel(x = as.numeric(xk), integration.points = U,
      integration.weights = NULL, intpoints.oldmean = pred$mean, intpoints.oldsd = pred$sd,
      precalc.data = precalc, model = model, T = 0, new.noise.var = NULL, batchsize = 1,
      current.sur = current.sur, ai_precalc = NULL)
}
w <- function(obj, name) write.table(obj, file.path(out, name), sep = ",", row.names = FALSE, col.names = FALSE)
w(X, "X.csv"); w(y, "y.csv"); w(U, "U.csv"); w(Xc, "Xc.csv")
w(cbind(pred$mean, pred$sd), "post_U.csv"); w(KN, "KN.csv"); w(sx2, "sx2.csv")
w(c(current.sur, crit), "crit.csv"); w(c(theta, sig2, mu0, nug), "params.csv")
cat("KrigInv", as.character(packageVersion("KrigInv")), "DiceKriging", as.character(packageVersion("DiceKriging")),
    "| H_n =", current.sur, "| crit range", range(crit), "\n")
