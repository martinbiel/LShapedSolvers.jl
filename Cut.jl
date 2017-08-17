abstract type AbstractCut end

proper(cut::AbstractCut) = true

struct OptimalityCut
    E::AbstractVector
    e::Real
    id::Integer
end

function OptimalityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Optimal "Trying to generate optimality cut from non-optimal subproblem"
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

    return OptimalityCut(E, e, subprob.id)
end

function addCut!(lshaped::AbstractLShapedSolver,cut::OptimalityCut)
    m = lshaped.masterModel
    cutIdx = m.numCols-nscenarios(lshaped)
    x = lshaped.structuredModel.colVal

    E = cut.E
    e = cut.e

    w = e-E⋅x
    θ = lshaped.ready[cut.id] ? getvalue(lshaped.θs[cut.id]) : -Inf
    τ = lshaped.τ

    lshaped.ready[cut.id] = true

    if abs((w-θ)/θ) <= τ
        # Optimal with respect to this subproblem
        println("θ",cut.id,": ", θ)
        println("w",cut.id,": ", w)
        println("Optimal with respect to subproblem ", cut.id)
        return false
    end

    # Add optimality cut
    @constraint(m,sum(E[i]*Variable(m,i)
                      for i = 1:cutIdx) + Variable(m,cutIdx+cut.id) >= e)
    lshaped.numOptimalityCuts += 1
    println("θ",cut.id,": ", θ)
    println("w",cut.id,": ", w)
    if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
        println("Added Optimality Cut")
    else
        println("Added Optimality Cut: ", lshaped.masterModel.linconstr[end])
    end
    return true
end

struct FeasibilityCut
    D::AbstractVector
    d::Real
end

function FeasibilityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Infeasible "Trying to generate feasibility cut from non-infeasible subproblem"
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

    return FeasibilityCut(D, d)
end

function addCut!(lshaped::AbstractLShapedSolver,cut::FeasibilityCut)
    m = lshaped.masterModel
    cutIdx = m.numCols-nscenarios(lshaped)

    D = cut.D
    d = cut.d

    # Scale to avoid numerical issues
    scaling = abs(d)
    if scaling == 0
        scaling = maximum(D)
    end

    D = D/scaling

    # Add feasibility cut
    @constraint(m,sum(D[i]*Variable(m,i)
                      for i = 1:cutIdx) >= sign(d))
    lshaped.numFeasibilityCuts += 1
    if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
        println("Added Feasibility Cut")
    else
        println("Added Feasibility Cut: ", lshaped.masterModel.linconstr[end])
    end
    return true
end

struct ImproperCut <: AbstractCut end
proper(cut::ImproperCut) = false
