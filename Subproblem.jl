mutable struct SubProblem
    model::JuMPModel
    parent::JuMPModel
    id::Integer
    π::Float64
    problem::LPProblem
    solver::LPSolver

    hl
    hu
    masterTerms

    function SubProblem(m::JuMPModel,parent::JuMPModel,id::Integer,π::Float64)
        subprob = new(m,parent,id,π)

        p = LPProblem(m)
        subprob.problem = p
        subprob.solver = LPSolver(p)
        subprob.hl = copy(p.l)
        subprob.hu = copy(p.u)

        subprob.masterTerms = []
        parseSubProblem!(subprob)

        return subprob
    end
end

function parseSubProblem!(subprob::SubProblem)
    for (i,constr) in enumerate(subprob.model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == subprob.parent
                # var is a first stage variable
                push!(subprob.masterTerms,(i,var,-constr.terms.coeffs[j]))
            end
        end
    end
end

function updateSubProblem!(subprob::SubProblem)
    rhsupdate = Dict{Int64,Float64}()
    for (i,x,coeff) in subprob.masterTerms
        if !haskey(rhsupdate,i)
            rhsupdate[i] = 0
        end
        rhsupdate[i] += coeff*getvalue(x)
    end

    numCols = subprob.problem.numCols

    for (i,rhs) in rhsupdate
        constr = subprob.model.linconstr[i]
        subprob.problem.l[numCols+i] = -(constr.ub + rhs)
        subprob.problem.u[numCols+i] = -(constr.lb + rhs)
    end
end

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
    E = zeros(subprob.parent.numCols)

    e = π*v[finite_hl]⋅hl[finite_hl] + π*w[finite_hu]⋅hu[finite_hu]

    for (i,x,coeff) in subprob.masterTerms
        E[x.col] += π*λ[i]*(-coeff)
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
    D = zeros(subprob.parent.numCols)

    d = v[finite_hl]⋅hl[finite_hl] + w[finite_hu]⋅hu[finite_hu]

    for (i,x,coeff) in subprob.masterTerms
        D[x.col] += λ[i]*(-coeff)
    end

    return D, d
end
