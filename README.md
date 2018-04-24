# Exact Matrix Completion via Convex Optimization(exact-mc)

- This repository reproduce experiments in the paper, [Exact Matrix Completion via Convex Optimization](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf).
  - Candès, Emmanuel J., and Benjamin Recht. "Exact matrix completion via convex optimization." Foundations of Computational mathematics 9.6 (2009): 717.
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
RANK of M    : 2
#sample      : 320
NUC-NORM of M: 36.76954064144393
STATUS       : optimal
OPTIMAL VALUE: 73.5256314253263
RANK of X    : 20
NUC-NORM of X: 36.76281571725943
TRAIN RMSE   : 6.626737339013724e-18
TEST  RMSE   : 0.006495179200041315
|X-M|_F/|M|_F: 0.019961863829646286
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
