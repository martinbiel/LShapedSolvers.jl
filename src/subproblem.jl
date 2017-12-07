mutable struct SubProblem
    id::Integer
    π::Real

    solver::AbstractLQSolver

    h::Tuple{AbstractVector,AbstractVector}
    x::AbstractVector
    masterTerms::AbstractVector

    function SubProblem(m::JuMPModel,parent::JuMPModel,id::Integer,π::Float64)
        subprob = new(id,π)

        subprob.solver = LQSolver(m)

        subprob.h = (getconstrLB(subprob.solver.model),getconstrUB(subprob.solver.model))
        subprob.x = zeros(parent.numCols)
        subprob.masterTerms = []
        parseSubProblem!(subprob,m,parent)

        return subprob
    end
end

function parseSubProblem!(subprob::SubProblem,model::JuMPModel,parent::JuMPModel)
    for (i,constr) in enumerate(model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == parent
                # var is a first stage variable
                push!(subprob.masterTerms,(i,var.col,-constr.terms.coeffs[j]))
            end
        end
    end
end

function updateSubProblem!(subprob::SubProblem,x::AbstractVector)
    lb = getconstrLB(subprob.solver.model)
    ub = getconstrUB(subprob.solver.model)
    for i in [term[1] for term in unique(term -> term[1],subprob.masterTerms)]
        lb[i] = subprob.h[1][i]
        ub[i] = subprob.h[2][i]
    end
    for (i,j,coeff) in subprob.masterTerms
        lb[i] += coeff*x[j]
        ub[i] += coeff*x[j]
    end
    setconstrLB!(subprob.solver.model, lb)
    setconstrUB!(subprob.solver.model, ub)
    subprob.x = x
end
updateSubProblems!(subprobs::Vector{SubProblem},x::AbstractVector) = map(prob -> updateSubProblem!(prob,x),subprobs)

function (subprob::SubProblem)()
    subprob.solver()
    solvestatus = status(subprob.solver)
    if solvestatus == :Optimal
        return OptimalityCut(subprob)
    elseif solvestatus == :Infeasible
        return FeasibilityCut(subprob)
    elseif lshaped.status == :Unbounded
        return Unbounded(subprob)
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subprob.id,string(solvestatus)))
    end
end

function (subprob::SubProblem)(x::AbstractVector)
    updateSubProblem!(subprob,x)
    subprob.solver()
    solvestatus = status(subprob.solver)
    if solvestatus == :Optimal
        return subprob.solver.obj
    elseif solvestatus == :Infeasible
        error(@sprintf("Subproblem %d is infeasible at the given first-stage variable",subprob.id))
    elseif lshaped.status == :Unbounded
        error(@sprintf("Subproblem %d is unbounded at the given first-stage variable",subprob.id))
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subprob.id,string(solvestatus)))
    end
end
