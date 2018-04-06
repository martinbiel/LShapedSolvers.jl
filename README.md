# LShaped

[![Build Status](https://travis-ci.org/martinbiel/LShaped.jl.svg?branch=master)](https://travis-ci.org/martinbiel/LShaped.jl)

[![Coverage Status](https://coveralls.io/repos/martinbiel/LShaped.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/martinbiel/LShaped.jl?branch=master)

[![codecov.io](http://codecov.io/github/martinbiel/LShaped.jl/coverage.svg?branch=master)](http://codecov.io/github/martinbiel/LShaped.jl?branch=master)

`LShapedSolvers` is a collection of structured optimization algorithms for two-stage (L-shaped) stochastic recourse problems. All algorithm variants are based on the L-shaped method by Van Slyke and Wets. `LShapedSolvers` interfaces with [StochasticPrograms.jl][StochProg], and a given recourse model `sp` is solved effectively through

```julia
julia> using LShapedSolvers

julia> solve(sp,solver=LShapedSolver(ClpSolver()))
:Optimal

```

Note, that an LP capable `AbstractMathProgSolver` is required to solve emerging subproblems. The following variants of the L-shaped algorithm are implemented:

1. L-shaped with multiple cuts (default): `LShapedSolver(:ls)`
2. L-shaped with regularized decomposition: `LShapedSolver(:rd)`
3. L-shaped with trust region: `LShapedSolver(:tr)`
4. L-shaped with level sets: `LShapedSolver(:lv)`

Note, that `LShapedSolver(:rd)` and `LShapedSolver(:lv)` both require a QP capable `AbstractMathProgSolver` for the master problems.

In addition, there is a distributed variant of each algorithm:

1. Distributed L-shaped with multiple cuts: `LShapedSolver(:dls)`
2. Distributed L-shaped with regularized decomposition: `LShapedSolver(:drd)`
3. Distributed L-shaped with trust region: `LShapedSolver(:dtr)`
4. Distributed L-shaped with level sets: `LShapedSolver(:dlv)`

which requires adding processes with `addprocs` prior to execution.

[StochProg]: https://github.com/martinbiel/StochasticPrograms.jl

## References

1. Van Slyke, R. and Wets, R. (1969), [L-Shaped Linear Programs with Applications to Optimal Control and Stochastic Programming](https://epubs.siam.org/doi/abs/10.1137/0117061),
SIAM Journal on Applied Mathematics, vol. 17, no. 4, pp. 638-663.

2. Ruszczyński, A (1986), [A regularized decomposition method for minimizing a sum of polyhedral functions](https://link.springer.com/article/10.1007/BF01580883),
Mathematical Programming, vol. 35, no. 3, pp. 309-333.

3. Linderoth, J. and Wright, S. (2003), [Decomposition Algorithms for Stochastic Programming on a Computational Grid](https://link.springer.com/article/10.1023/A:1021858008222),
Computational Optimization and Applications, vol. 24, no. 2-3, pp. 207-250.

4. F{\'{a}}bi{\'{a}}n, C. and Sz{\H{o}}ke, Z. (2006), [Solving two-stage stochastic programming problems with level decomposition](https://link.springer.com/article/10.1007%2Fs10287-006-0026-8),
Computational Management Science, vol. 4, no. 4, pp. 313-353.
