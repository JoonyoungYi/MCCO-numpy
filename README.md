# Exact Matrix Completion via Convex Optimization(exact-mc)

- This repository reproduce experiments in the paper, [Exact Matrix Completion via Convex Optimization](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf).
  - Candès, Emmanuel J., and Benjamin Recht. "Exact matrix completion via convex optimization." Foundations of Computational mathematics 9.6 (2009): 717.
- How to install and run.
```
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```
- I implemented solution of convex optimization(nuclear norm minimization) via SDP(semidefinite programming).
- The idea of process reducing from nuclear norm minimization to SDP is from \[Fazel et al. 2002\](The paper already mentioned).
- The meanings of parameters.
```
n_1: The number of row of the matrix
n_2: The number of column of the matrix
r: The rank of the matrix.
M: The matrix to recover.
X': The matrix to recover in SDP.
X: The solution of nuclear norm minimization.
```
