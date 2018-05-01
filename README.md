# Exact Matrix Completion via Convex Optimization(MCCO-numpy)

- This repository reproduce experiments in the paper, [Exact Matrix Completion via Convex Optimization](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf).
  - Candès, Emmanuel J., and Benjamin Recht. "Exact matrix completion via convex optimization." Foundations of Computational mathematics 9.6 (2009): 717.
  - I made [slide](https://www.slideshare.net/ssuser62b35f/exact-matrix-completion-via-convex-optimization-slideppt) of this paper. You can see the [slide](https://www.slideshare.net/ssuser62b35f/exact-matrix-completion-via-convex-optimization-slideppt) for better understanding.
- I try to minimize the number of packages to install. As the result of this effort, you can use this repository by only installing several packages only with python virtual environment.
- How to install and run. I've tested on python3.6. But, I think python3 is ok.
```
virtualenv .venv -p python3.6
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```
- The example of result(console)
```
#row of M    : 20
#column of M : 20
#sample      : 320
|M|_*        : 33.49115055002033
RANK of M    : 2
[2.10261945e+01 1.24649561e+01 2.98926553e-15 1.01018956e-15
 8.08684201e-16 6.52764904e-16 5.43264821e-16 4.52477775e-16
 3.82599383e-16 2.96536152e-16 2.78437945e-16 1.95651603e-16
 1.79018550e-16 1.51585299e-16 9.25091603e-17 8.09000656e-17
 6.40724175e-17 2.84048051e-17 1.22531514e-17 2.35572616e-18]
STATUS       : optimal
OPTIMAL VALUE: 66.9823010865963
|X0-X0.T|_F  : 0.0
RANK of X    : 20
|X|_*        : 33.49115056009586
[2.10261945e+01 1.24649561e+01 5.23798589e-09 3.17159076e-09
 1.96692896e-09 1.36585962e-09 1.23291953e-09 9.43323942e-10
 6.98082407e-10 5.75897347e-10 5.33940684e-10 5.08809987e-10
 2.84172525e-10 2.11719465e-10 1.95427315e-10 1.25889321e-10
 7.65764295e-11 4.34220283e-11 3.69422261e-12 1.81272589e-13]
TRAIN RMSE   : 7.298183225902274e-18
TEST  RMSE   : 1.47376828185977e-10
|X-M|_F/|M|_F: 4.823463160263553e-10
```
- I implemented the solution of convex optimization(nuclear norm minimization) via SDP(semidefinite programming).
- The idea of process reducing from nuclear norm minimization to SDP is from \[Fazel et al. 2002\](The paper already mentioned).
- The meanings of parameters.
```
M: The true matrix.
n_1: The number of row of the matrix M.
n_2: The number of column of the matrix M.
r: The rank of the matrix M.
omega: The sample set. observed entry (i, j) in the sample set omega.
M_: Input of algorithm. Th ematrix to recover.
X_: The matrix to recover in SDP.
X: The solution of nuclear norm minimization.
```
- You can optimize more to enhance performance.
- This paper handles the noise-less case. If your data has noise, you should add one or more regularizer term to objective function. See [cvxpy](http://www.cvxpy.org/en/latest/).
