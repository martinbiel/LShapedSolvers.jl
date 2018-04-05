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
    fill_subproblems!(lshaped,scenarioproblems(stochasticprogram))
end

function fill_subproblems!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems)
    for (i,submodel) in enumerate(scenarioproblems.problems)
        snrows, sncols = length(submodel.linconstr), submodel.numCols
        subproblem = lshaped.subproblems[i]
        submodel.colVal = copy(subproblem.y)
        submodel.redCosts = try
            getreducedcosts(subproblem.solver.lqmodel)[1:sncols]
        catch
            fill(NaN, sncols)
        end
        submodel.linconstrDuals = try
            getconstrduals(subproblem.solver.lqmodel)[1:snrows]
        catch
            fill(NaN, snrows)
        end
        submodel.objVal = getobjval(subproblem.solver)
    end
end

function fill_subproblems!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems)
    finished_workers = Vector{Future}(length(scenarioproblems))
    for w = 1:length(scenarioproblems)
        finished_workers[w] = remotecall(fill_subproblems!,
                                         w+1,
                                         lshaped.subworkers[w],
                                         scenarioproblems[w])
    end
    map(wait,finished_workers)
end
