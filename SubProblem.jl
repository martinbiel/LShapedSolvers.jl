mutable struct SubProblem
    id::Integer
    π::Real

    solver::AbstractLQSolver

    hlb::AbstractVector
    hub::AbstractVector
    x::AbstractVector
    masterTerms::AbstractVector

    function SubProblem(m::JuMPModel,parent::JuMPModel,id::Integer,π::Float64)
        subprob = new(id,π)

        subprob.solver = LQSolver(m)

        subprob.hlb = getconstrLB(subprob.solver.model)
        subprob.hub = getconstrUB(subprob.solver.model)
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
        lb[i] = subprob.hlb[i]
        ub[i] = subprob.hub[i]
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

function getOptimalityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Optimal "Trying to generate optimality cut from non-optimal subproblem"
    λ = subprob.solver.λ
    hl = subprob.hl
    hu = subprob.hu
    π = subprob.π

    cols = zeros(length(subprob.masterTerms))
    vals = zeros(length(subprob.masterTerms))
    for (s,(i,j,coeff)) in enumerate(subprob.masterTerms)
        cols[s] = j
        vals[s] = -π*λ[i]*coeff
    end
    δQ = sparsevec(cols,vals,subprob.nMasterCols)
    q = subprob.solver.obj - δQ⋅subprob.x

    return OptimalityCut(δQ, q, subprob.id)
end

function getFeasibilityCut(subprob::SubProblem)
    # @assert status(subprob.solver) == :Infeasible
    # λ = subprob.solver.λ
    # v = subprob.solver.v
    # w = subprob.solver.w
    # hl = subprob.hl
    # hu = subprob.hu
    # finite_hl = find(!isinf,hl)
    # finite_hu = find(!isinf,hu)
    # D = zeros(subprob.nMasterCols)

    # d = v[finite_hl]⋅hl[finite_hl] + w[finite_hu]⋅hu[finite_hu]

    # for (i,j,coeff) in subprob.masterTerms
    #     D[j] -= λ[i]*coeff
    # end

    # return D, d
end

function (subprob::SubProblem)()
    subprob.solver()
    solvestatus = status(subprob.solver)
    if solvestatus == :Optimal
        return OptimalityCut(subprob)
    elseif solvestatus == :Infeasible
        return FeasibilityCut(subprob)
    elseif lshaped.status == :Unbounded
        return ImproperCut(subprob)
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
