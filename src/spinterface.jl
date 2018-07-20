struct LShapedSolver <: AbstractStructuredSolver
    variant::Symbol
    lpsolver::AbstractMathProgSolver
    subsolver::AbstractMathProgSolver
    crash::Crash.CrashMethod
    parameters

    function (::Type{LShapedSolver})(variant::Symbol, lpsolver::AbstractMathProgSolver; crash::Crash.CrashMethod = Crash.None(), subsolver = lpsolver, kwargs...)
        return new(variant,lpsolver,subsolver,crash,kwargs)
    end
end
LShapedSolver(lpsolver::AbstractMathProgSolver; kwargs...) = LShapedSolver(:ls, lpsolver, kwargs...)

function StructuredModel(solver::LShapedSolver,stochasticprogram::JuMP.Model)
    x₀ = solver.crash(stochasticprogram,solver.lpsolver)
    if solver.variant == :ls
        return LShaped(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :dls
        return DLShaped(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :rd
        return Regularized(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :drd
        return DRegularized(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :tr
        return TrustRegion(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :dtr
        return DTrustRegion(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :lv
        return LevelSet(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :dlv
        return DLevelSet(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    elseif solver.variant == :llv
        return LinearLevelSet(stochasticprogram,x₀,solver.lpsolver,solver.subsolver; solver.parameters...)
    else
        error("Unknown L-Shaped variant: ", solver.variant)
    end
end

function optimsolver(solver::LShapedSolver)
    return solver.lpsolver
end

function optimize_structured!(lshaped::AbstractLShapedSolver)
    return lshaped()
end

function fill_solution!(lshaped::AbstractLShapedSolver,stochasticprogram::JuMP.Model)
    # First stage
    nrows, ncols = length(stochasticprogram.linconstr), stochasticprogram.numCols
    stochasticprogram.colVal = copy(lshaped.x)
    stochasticprogram.redCosts = try
        getreducedcosts(lshaped.mastersolver.lqmodel)[1:ncols]
    catch
        fill(NaN, ncols)
    end
    stochasticprogram.linconstrDuals = try
        getconstrduals(lshaped.mastersolver.lqmodel)[1:nrows]
    catch
        fill(NaN, nrows)
    end
    # Second stage
    fill_submodels!(lshaped,scenarioproblems(stochasticprogram))
    # Now safe to generate the objective value of the stochastic program
    stochasticprogram.objVal = StochasticPrograms.calculate_objective_value(stochasticprogram)
end
