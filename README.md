# LShapedSolvers

[![Build Status](https://travis-ci.org/martinbiel/LShapedSolvers.jl.svg?branch=test)](https://travis-ci.org/martinbiel/LShapedSolvers.jl)

[![Coverage Status](https://coveralls.io/repos/martinbiel/LShapedSolvers.jl/badge.svg?branch=test&service=github)](https://coveralls.io/github/martinbiel/LShaped.jl?branch=test)

[![codecov.io](http://codecov.io/github/martinbiel/LShapedSolvers.jl/coverage.svg?branch=test)](http://codecov.io/github/martinbiel/LShapedSolvers.jl?branch=test)

`LShapedSolvers` is a collection of structured optimization algorithms for two-stage (L-shaped) stochastic recourse problems. All algorithm variants are based on the L-shaped method by Van Slyke and Wets. `LShapedSolvers` interfaces with [StochasticPrograms.jl][StochProg], and a given recourse model `sp` is solved effectively through

```julia
julia> using LShapedSolvers

julia> solve(sp,solver=LShapedSolver(ClpSolver()))
L-Shaped Gap  Time: 0:00:01 (4 iterations)
  Objective:       -855.8333333333358
  Gap:             2.1229209144670507e-15
  Number of cuts:  5
:Optimal

```

Note, that an LP capable `AbstractMathProgSolver` is required to solve emerging subproblems. Solver objects are obtained through the factory method `LShapedSolver`. The following variants of the L-shaped algorithm are implemented:

1. L-shaped with multiple cuts (default): `regularization = :none (default)`
2. L-shaped with regularized decomposition: `regularization = :rd`
3. L-shaped with trust region: `regularization = :tr`
4. L-shaped with level sets: `regularization = :lv`

Note, that `:rd` and `:lv` both require a QP capable `AbstractMathProgSolver` for the master problems. If not available, setting the `linearize` keyword to `true` is an alternative.

In addition, there is a distributed variant of each algorithm, which requires adding processes with `addprocs` prior to execution. The distributed variants are obtained by supplying `distributed = true` to `LShapedSolver`.

Each algorithm has a set of parameters that can be tuned prior to execution. For a list of these parameters and their default values, use `?` in combination with the solver object. For example, `?LShaped` gives the parameter list of the default L-shaped algorithm. For a list of all solvers and their handle names, use `?LShapedSolver`.

`LShapedSolvers.jl` includes a set of crash methods that can be used to generate the initial decision by supplying functor objects to `LShapedSolver`. Use `?Crash` for a list of available crashes and their usage.

[StochProg]: https://github.com/martinbiel/StochasticPrograms.jl

## References

1. Van Slyke, R. and Wets, R. (1969), [L-Shaped Linear Programs with Applications to Optimal Control and Stochastic Programming](https://epubs.siam.org/doi/abs/10.1137/0117061),
SIAM Journal on Applied Mathematics, vol. 17, no. 4, pp. 638-663.

2. Ruszczyński, A (1986), [A regularized decomposition method for minimizing a sum of polyhedral functions](https://link.springer.com/article/10.1007/BF01580883),
Mathematical Programming, vol. 35, no. 3, pp. 309-333.

3. Linderoth, J. and Wright, S. (2003), [Decomposition Algorithms for Stochastic Programming on a Computational Grid](https://link.springer.com/article/10.1023/A:1021858008222),
Computational Optimization and Applications, vol. 24, no. 2-3, pp. 207-250.

4. Fábián, C. and Szőke, Z. (2006), [Solving two-stage stochastic programming problems with level decomposition](https://link.springer.com/article/10.1007%2Fs10287-006-0026-8),
Computational Management Science, vol. 4, no. 4, pp. 313-353.
