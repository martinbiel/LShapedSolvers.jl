struct LShapedSolver{S <: AbstractMathProgSolver} <: AbstractStructuredSolver
    variant::Symbol
    lpsolver::S
    parameters

    function (::Type{LShapedSolver})(variant::Symbol, lpsolver::AbstractMathProgSolver; kwargs...)
        return new{typeof(lpsolver)}(variant,lpsolver,kwargs)
    end
end
LShapedSolver(lpsolver::AbstractMathProgSolver; kwargs...) = LShapedSolver(:ls, lpsolver, kwargs...)

function StructuredModel(solver::LShapedSolver,stochasticprogram::JuMP.Model; crash::Crash.CrashMethod = Crash.None(), subsolver = solver.lpsolver)
    x₀ = crash(stochasticprogram,solver.lpsolver)
    if solver.variant == :ls
        return LShaped(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :dls
        return DLShaped(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :rd
        return Regularized(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :drd
        return DRegularized(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :tr
        return TrustRegion(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :dtr
        return DTrustRegion(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :lv
        return LevelSet(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    elseif solver.variant == :dlv
        return DLevelSet(stochasticprogram,x₀,solver.lpsolver,subsolver; solver.parameters...)
    else
        error("Unknown L-Shaped variant: ", solver.variant)
    end
end

function optimize_structured!(lshaped::AbstractLShapedSolver)
    return lshaped()
end

function fill_solution!(lshaped::AbstractLShapedSolver,stochasticprogram::JuMP.Model)
    # First stage
    nrows, ncols = length(stochasticprogram.linconstr), stochasticprogram.numCols
    stochasticprogram.objVal = lshaped.solverdata.Q
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
end
