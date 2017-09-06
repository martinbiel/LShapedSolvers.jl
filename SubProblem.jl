mutable struct SubProblem
    model::JuMPModel
    id::Integer
    π::Real

    problem::LPProblem
    solver::LPSolver

    hl::AbstractVector
    hu::AbstractVector
    nMasterCols::Integer
    masterTerms::AbstractVector

    function SubProblem(m::JuMPModel,parent::JuMPModel,id::Integer,π::Float64)
        subprob = new(m,id,π)

        p = LPProblem(m)
        subprob.problem = p
        subprob.solver = LPSolver(p)

        subprob.hl = copy(p.l)
        subprob.hu = copy(p.u)
        subprob.nMasterCols = parent.numCols
        subprob.masterTerms = []
        parseSubProblem!(subprob,parent)

        return subprob
    end
end

function parseSubProblem!(subprob::SubProblem,parent::JuMPModel)
    for (i,constr) in enumerate(subprob.model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == parent
                # var is a first stage variable
                push!(subprob.masterTerms,(i,var.col,-constr.terms.coeffs[j]))
            end
        end
    end
end

function updateSubProblem!(subprob::SubProblem,x::AbstractVector)
    m = subprob.problem.numCols
    for i in [term[1] for term in unique(term -> term[1],subprob.masterTerms)]
        constr = subprob.model.linconstr[i]
        subprob.problem.l[m+i] = -constr.ub
        subprob.problem.u[m+i] = -constr.lb
    end
    for (i,j,coeff) in subprob.masterTerms
        subprob.problem.l[m+i] -= coeff*x[j]
        subprob.problem.u[m+i] -= coeff*x[j]
    end
    setvarLB!(subprob.solver.model, subprob.problem.l)
    setvarUB!(subprob.solver.model, subprob.problem.u)
end

updateSubProblems!(subprobs::Vector{SubProblem},x::AbstractVector) = map(prob -> updateSubProblem!(prob,x),subprobs)

function getOptimalityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Optimal
    λ = subprob.solver.λ
    v = subprob.solver.v
    w = subprob.solver.w
    hl = subprob.hl
    hu = subprob.hu
    finite_hl = find(!isinf,hl)
    finite_hu = find(!isinf,hu)
    π = subprob.π
    E = zeros(subprob.nMasterCols)

    e = π*v[finite_hl]⋅hl[finite_hl] + π*w[finite_hu]⋅hu[finite_hu]

    for (i,j,coeff) in subprob.masterTerms
        E[j] -= π*λ[i]*coeff
    end

    return E, e
end

function getFeasibilityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Infeasible
    λ = subprob.solver.λ
    v = subprob.solver.v
    w = subprob.solver.w
    hl = subprob.hl
    hu = subprob.hu
    finite_hl = find(!isinf,hl)
    finite_hu = find(!isinf,hu)
    D = zeros(subprob.nMasterCols)

    d = v[finite_hl]⋅hl[finite_hl] + w[finite_hu]⋅hu[finite_hu]

    for (i,j,coeff) in subprob.masterTerms
        D[j] -= λ[i]*coeff
    end

    return D, d
end

function (subprob::SubProblem)()
    subprob.solver()
    solvestatus = status(subprob.solver)
    if solvestatus == :Optimal
        updateSolution(subprob.solver,subprob.model)
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
