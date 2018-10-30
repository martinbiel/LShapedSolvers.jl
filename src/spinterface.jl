"""
    LShapedSolver(variant::Symbol = :ls, lpsolver::AbstractMathProgSolver; <keyword arguments>)

Return the L-shaped algorithm object specified by the `variant` symbol. Supply `lpsolver`, a MathProgBase solver capable of solving linear quadratic problems.

The available algorithm variants are as follows
- `:ls`:  L-shaped algorithm (default) ?LShaped for parameter descriptions.
- `:dls`: Distributed L-shaped (requires worker cores) ?DLShaped for parameter descriptions.
- `:rd`:  Regularized decomposition ?Regularized for parameter descriptions.
- `:drd`: Distributed regularized (requires worker cores) ?DRegularized for parameter descriptions.
- `:tr`:  Trust-region ?TrustRegion for parameter descriptions.
- `:dtr`: Distributed trust-region (requires worker cores) ?DTrustRegion for parameter descriptions.
- `:lv`:  Level-set ?LevelSet for parameter descriptions.
- `:dlv`: Distributed level-set (requires worker cores) ?DLevelSet for parameter descriptions.

...
# Arguments
- `variant::Symbol = :ls`: L-shaped algorithm variant.
- `lpsolver::AbstractMathProgSolver`: MathProgBase solver capable of solving linear (and possibly quadratic) programs.
- `subsolver::AbstractMathProgSolver = lpsolver`: Optionally specify a different solver for the subproblems.
- `projectionsolver::AbstractMathProgSolver = lpsolver`: Optionally specify a different solver for solving projection problems (only applies in level-set variants).
- `checkfeas::Bool = false`: Specify if feasibility cuts should be used or not. (Should be false if problem is known to have (relatively) complete recourse for best performance.)
- `crash::Crash.CrashMethod = Crash.None`: Crash method used to generate an initial decision. See ?Crash for alternatives.
- <keyword arguments>: Algorithm specific parameters, consult individual docstrings (see above list) for list of possible arguments and default values.
...

## Examples

The following solves a stochastic program `sp` created in `StochasticPrograms.jl` using the L-shaped algorithm with Clp as an `lpsolver`.

```jldoctest
julia> solve(sp,solver=LShapedSolver(:ls,GLPKSolverLP()))
L-Shaped Gap  Time: 0:00:01 (4 iterations)
  Objective:       -855.8333333333339
  Gap:             0.0
  Number of cuts:  7
:Optimal
```
"""
mutable struct LShapedSolver <: AbstractStructuredSolver
    variant::Symbol
    lpsolver::MPB.AbstractMathProgSolver
    subsolver::MPB.AbstractMathProgSolver
    projectionsolver::MPB.AbstractMathProgSolver
    checkfeas::Bool
    crash::Crash.CrashMethod
    parameters::Dict{Symbol,Any}

    function (::Type{LShapedSolver})(variant::Symbol, lpsolver::MPB.AbstractMathProgSolver; crash::Crash.CrashMethod = Crash.None(), subsolver::MPB.AbstractMathProgSolver = lpsolver, projectionsolver::MPB.AbstractMathProgSolver = lpsolver, checkfeas::Bool = false, kwargs...)
        return new(variant, lpsolver, subsolver, projectionsolver, checkfeas, crash, Dict{Symbol,Any}(kwargs))
    end
end
LShapedSolver(lpsolver::MPB.AbstractMathProgSolver; kwargs...) = LShapedSolver(:ls, lpsolver, kwargs...)

function StructuredModel(stochasticprogram::StochasticProgram, solver::LShapedSolver)
    x₀ = solver.crash(stochasticprogram, solver.lpsolver)
    if solver.variant == :ls
        return LShaped(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :dls
        return DLShaped(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :rd
        return Regularized(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :drd
        return DRegularized(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :tr
        return TrustRegion(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :dtr
        return DTrustRegion(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :lv
        return LevelSet(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.projectionsolver, solver.checkfeas; solver.parameters...)
    elseif solver.variant == :dlv
        return DLevelSet(stochasticprogram, x₀, solver.lpsolver, solver.subsolver, solver.projectionsolver, solver.checkfeas; solver.parameters...)
    else
        error("Unknown L-Shaped variant: ", solver.variant)
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
