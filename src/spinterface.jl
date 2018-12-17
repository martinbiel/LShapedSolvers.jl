"""
    LShapedSolver(lpsolver::AbstractMathProgSolver; <keyword arguments>)

Return an L-shaped algorithm object. Supply `lpsolver`, a MathProgBase solver capable of solving linear-quadratic problems.

The available L-shaped regularizations are as follows
- `:none`:  L-shaped algorithm (default) ?LShaped for parameter descriptions.
- `:rd`:  Regularized decomposition ?Regularized for parameter descriptions.
- `:tr`:  Trust-region ?TrustRegion for parameter descriptions.
- `:lv`:  Level-set ?LevelSet for parameter descriptions.

...
# Arguments
- `variant::Symbol = :ls`: L-shaped algorithm variant.
- `lpsolver::AbstractMathProgSolver`: MathProgBase solver capable of solving linear (and possibly quadratic) programs.
- `subsolver::AbstractMathProgSolver = lpsolver`: Optionally specify a different solver for the subproblems.
- `projectionsolver::AbstractMathProgSolver = lpsolver`: Optionally specify a different solver for solving projection problems (only applies in level-set variants).
- `checkfeas::Bool = false`: Specify if feasibility cuts should be used or not. (Should be false if problem is known to have (relatively) complete recourse for best performance.)
- `regularization::Symbol = :none`: Specify regularization procedure (:none, :rd, :tr, :lv).
- `distributed::Bool = false`: Specify if distributed variant of algorithm should be run (requires worker cores). See `?Alg` for parameter descriptions.
- `crash::Crash.CrashMethod = Crash.None`: Crash method used to generate an initial decision. See ?Crash for alternatives.
- <keyword arguments>: Algorithm specific parameters, consult individual docstrings (see above list) for list of possible arguments and default values.
...

## Examples

The following solves a stochastic program `sp` created in `StochasticPrograms.jl` using the L-shaped algorithm with GLPK as an `lpsolver`.

```jldoctest
julia> solve(sp,solver=LShapedSolver(GLPKSolverLP()))
L-Shaped Gap  Time: 0:00:01 (4 iterations)
  Objective:       -855.8333333333339
  Gap:             0.0
  Number of cuts:  7
:Optimal
```
"""
mutable struct LShapedSolver <: AbstractStructuredSolver
    lpsolver::MPB.AbstractMathProgSolver
    subsolver::MPB.AbstractMathProgSolver
    projectionsolver::MPB.AbstractMathProgSolver
    checkfeas::Bool
    regularization::Symbol
    distributed::Bool
    crash::Crash.CrashMethod
    parameters::Dict{Symbol,Any}

    function (::Type{LShapedSolver})(lpsolver::MPB.AbstractMathProgSolver; checkfeas::Bool = false, regularization = :none, distributed = false, crash::Crash.CrashMethod = Crash.None(), subsolver::MPB.AbstractMathProgSolver = lpsolver, projectionsolver::MPB.AbstractMathProgSolver = lpsolver, kwargs...)
        return new(lpsolver, subsolver, projectionsolver, checkfeas, regularization, distributed, crash, Dict{Symbol,Any}(kwargs))
    end
end

function StructuredModel(stochasticprogram::StochasticProgram, solver::LShapedSolver)
    x₀ = solver.crash(stochasticprogram, solver.lpsolver)
    if solver.regularization == :none
        if solver.distributed
            return DLShaped(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        else
            return LShaped(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        end
    elseif solver.regularization == :rd
        if solver.distributed
            return DRegularized(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        else
            return Regularized(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        end
    elseif solver.regularization == :tr
        if solver.distributed
            return DTrustRegion(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        else
            return TrustRegion(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
        end
    elseif solver.regularization == :lv
        if solver.distributed
            return DLevelSet(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.projectionsolver, solver.checkfeas; solver.parameters...)
        else
            return LevelSet(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.projectionsolver, solver.checkfeas; solver.parameters...)
        end
    else
        error("Unknown L-shaped regularization procedure: ", solver.regularization)
    end
end

function add_params!(solver::LShapedSolver; kwargs...)
    push!(solver.parameters,kwargs...)
    for (k,v) in kwargs
        if k ∈ [:variant, :lpsolver, :subsolver, :projectionsolver, :checkfeas, :crash]
            setfield!(solver,k,v)
            delete!(solver.parameters, k)
        end
    end
end

function internal_solver(solver::LShapedSolver)
    return solver.lpsolver
end

function optimize_structured!(lshaped::AbstractLShapedSolver)
    return lshaped()
end

function fill_solution!(stochasticprogram::StochasticProgram, lshaped::AbstractLShapedSolver)
    # First stage
    first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
    nrows, ncols = first_stage_dims(stochasticprogram)
    StochasticPrograms.set_decision!(stochasticprogram, decision(lshaped))
    μ = try
        MPB.getreducedcosts(lshaped.mastersolver.lqmodel)[1:ncols]
    catch
        fill(NaN, ncols)
    end
    StochasticPrograms.set_first_stage_redcosts!(stochasticprogram, μ)
    λ = try
        MPB.getconstrduals(lshaped.mastersolver.lqmodel)[1:nrows]
    catch
        fill(NaN, nrows)
    end
    StochasticPrograms.set_first_stage_duals!(stochasticprogram, λ)
    # Second stage
    fill_submodels!(lshaped, scenarioproblems(stochasticprogram))
end

function solverstr(solver::LShapedSolver)
    if solver.variant == :ls
        return "L-shaped"
    elseif solver.variant == :dls
        return "Distributed L-shaped"
    elseif solver.variant == :rd
        return "Regularized L-shaped"
    elseif solver.variant == :drd
        return "Distributed regularized L-shaped"
    elseif solver.variant == :tr
        return "Trust-region L-shaped"
    elseif solver.variant == :dtr
        return "Distributed trust-region L-shaped"
    elseif solver.variant == :lv
        return "Level-set L-shaped"
    elseif solver.variant == :dlv
        return "Distributed level-set L-shaped"
    else
        error("Unknown L-Shaped variant: ", solver.variant)
    end
end
