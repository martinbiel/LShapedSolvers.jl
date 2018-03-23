struct LShapedSolver{S <: AbstractMathProgSolver} <: AbstractStructuredSolver
    variant::Symbol
    lpsolver::S
    parameters

    function (::Type{LShapedSolver})(variant::Symbol, lpsolver::AbstractMathProgSolver; kwargs...)
        return new{typeof(lpsolver)}(variant,lpsolver,kwargs)
    end
end
LShapedSolver(lpsolver::AbstractMathProgSolver; kwargs...) = LShapedSolver(:ls, lpsolver, kwargs...)

function StructuredModel(solver::LShapedSolver,stochasticprogram::JuMP.Model; crash=false, subsolver=solver.lpsolver)
    x₀ = if crash
        evp = StochasticPrograms.EVP(stochasticprogram,solver.lpsolver)
        status = solve(evp)
        status != :Optimal && error("Could not solve EVP model during crash procedure. Aborting.")
        evp.colVal[1:stochasticprogram.numCols]
    else
        rand(stochasticprogram.numCols)
    end
    if solver.variant == :ls
        return LShaped(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :als
        return ALShaped(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :rd
        return Regularized(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :ard
        return ARegularized(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :tr
        return TrustRegion(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :atr
        return ATrustRegion(stochasticprogram,x₀,solver.lpsolver,subsolver)
    elseif solver.variant == :lv
        return LevelSet(stochasticprogram,x₀,solver.lpsolver,subsolver)
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
    stochasticprogram.objVal = calculate_objective_value(lshaped)
    stochasticprogram.colVal = copy(lshaped.x)
    stochasticprogram.redCosts = getreducedcosts(lshaped.mastersolver.lqmodel)
    stochasticprogram.linconstrDuals = getconstrduals(lshaped.mastersolver.lqmodel)

    # Second stage
    for (i,submodel) in enumerate(subproblems(stochasticprogram))
        snrows, sncols = length(submodel.linconstr), submodel.numCols
        subproblem = lshaped.subproblems[i]
        submodel.colVal = copy(subproblem.y)
        submodel.redCosts = getreducedcosts(subproblem.solver.lqmodel)
        submodel.linconstrDuals = getconstrduals(subproblem.solver.lqmodel)
        submodel.objVal = getobjval(subproblem.solver)
    end
end
