abstract type AbstractHyperplane end

proper(cut::AbstractHyperplane) = true

function active(lshaped::AbstractLShapedSolver,hyperplane::AbstractHyperplane)
    Gval,g = hyperplane(lshaped.x)
    return abs(Gval-g) <= lshaped.τ*(1+abs(Gval))
end
function satisfied(lshaped::AbstractLShapedSolver,hyperplane::AbstractHyperplane)
    Gval,g = hyperplane(lshaped.x)
    return Gval >= g - lshaped.τ*(1+abs(Gval))
end
function violated(lshaped::AbstractLShapedSolver,hyperplane::AbstractHyperplane)
    return !satisfied(lshaped,hyperplane)
end
function gap(lshaped::AbstractLShapedSolver,hyperplane::AbstractHyperplane)
    Gval,g = hyperplane(lshaped.x)
    return Gval-g
end
function lowlevel(hyperplane::AbstractHyperplane)
    return hyperplane.G.nzind,hyperplane.G.nzval,hyperplane.g,Inf
end

struct LinearConstraint <: AbstractHyperplane
    G::AbstractVector
    g::Real
    id::Integer
end

function LinearConstraint(constraint::JuMP.LinearConstraint,i::Integer)
    sense = JuMP.sense(constraint)
    if sense == :range
        throw(ArgumentError("Cannot handle range constraints"))
    end
    cols = map(v->v.col,constraint.terms.vars)
    vals = constraint.terms.coeffs * (sense == :(>=) ? 1 : -1)
    G = sparsevec(cols,vals,constraint.terms.vars[1].m.numCols)
    g = JuMP.rhs(constraint) * (sense == :(>=) ? 1 : -1)

    return LinearConstraint(G,g,i)
end

function linearconstraints(m::JuMPModel)
    constraints = Vector{LinearConstraint}(length(m.linconstr))
    for (i,c) in enumerate(m.linconstr)
        constraints[i] = LinearConstraint(c,i)
    end
    return constraints
end

function (constraint::LinearConstraint)(x::AbstractVector)
    if length(constraint.G) != length(x)
        throw(ArgumentError(@sprintf("Dimensions of the cut (%d)) and the given optimization vector (%d) does not match",length(constraint.G),length(x))))
    end
    return constraint.G⋅x,constraint.g
end

struct OptimalityCut <: AbstractHyperplane
    δQ::AbstractVector
    q::Real
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

    q = π*v[finite_hl]⋅hl[finite_hl] + π*w[finite_hu]⋅hu[finite_hu]

    cols = zeros(length(subprob.masterTerms))
    vals = zeros(length(subprob.masterTerms))
    for (s,(i,j,coeff)) in enumerate(subprob.masterTerms)
        cols[s] = j
        vals[s] = -π*λ[i]*coeff
    end
    δQ = sparsevec(cols,vals,subprob.nMasterCols)

    return OptimalityCut(δQ, q, subprob.id)
end

function (cut::OptimalityCut)(x::AbstractVector)
    if length(cut.δQ) != length(x)
        throw(ArgumentError(@sprintf("Dimensions of the cut (%d)) and the given optimization vector (%d) does not match",length(cut.δQ),length(x))))
    end
    return cut.q-cut.δQ⋅x
end

function lowlevel(cut::OptimalityCut)
    nzind = copy(cut.δQ.nzind)
    nzval = copy(cut.δQ.nzval)
    push!(nzind,length(cut.δQ)+cut.id)
    push!(nzval,1.0)
    return nzind,nzval,cut.q,Inf
end

function addCut!(lshaped::AbstractLShapedSolver,cut::OptimalityCut)
    m = lshaped.masterModel
    cutIdx = m.numCols-nscenarios(lshaped)

    Q = cut(lshaped.x)
    θ = lshaped.ready[cut.id] ? getvalue(lshaped.θs[cut.id]) : -Inf
    τ = lshaped.τ

    lshaped.subObjectives[cut.id] = Q
    lshaped.ready[cut.id] = true

    println("θ",cut.id,": ", θ)
    println("Q",cut.id,": ", Q)

    if θ > -Inf && abs(θ-Q) <= τ*(1+abs(Q))
        # Optimal with respect to this subproblem
        println("Optimal with respect to subproblem ", cut.id)
        return false
    end

    # Add optimality cut
    @constraint(m,sum(cut.δQ[i]*Variable(m,i)
                      for i = 1:cutIdx) + Variable(m,cutIdx+cut.id) >= cut.q)
    lshaped.nOptimalityCuts += 1
    if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
        println("Added Optimality Cut")
    else
        println("Added Optimality Cut: ", lshaped.masterModel.linconstr[end])
    end
    if istrait(IsRegularized{typeof(lshaped)})
        push!(lshaped.committee,cut)
        addconstr!(lshaped.internal,lowlevel(cut)...)
    else
        addconstr!(lshaped.masterSolver.model,lowlevel(cut)...)
    end
    push!(lshaped.cuts,cut)
    return true
end

function optimal(lshaped::AbstractLShapedSolver,cut::OptimalityCut)
    Q = cut(lshaped.x)
    θ = lshaped.ready[cut.id] ? getvalue(lshaped.θs[cut.id]) : -Inf
    return θ > -Inf && abs(θ-Q) <= lshaped.τ*(1+abs(Q))
end
active(lshaped::AbstractLShapedSolver,cut::OptimalityCut) = optimal(lshaped,cut)

function satisfied(lshaped::AbstractLShapedSolver,cut::OptimalityCut)
    Q = cut(lshaped.x)
    θ = lshaped.ready[cut.id] ? getvalue(lshaped.θs[cut.id]) : -Inf
    return θ > -Inf && θ >= Q - lshaped.τ*(1+abs(Q))
end

function gap(lshaped::AbstractLShapedSolver,cut::OptimalityCut)
    Q = cut(lshaped.x)
    θ = lshaped.ready[cut.id] ? getvalue(lshaped.θs[cut.id]) : -Inf
    if θ > -Inf
        return θ-Q
    else
        return Inf
    end
end

struct FeasibilityCut <: AbstractHyperplane
    G::AbstractVector
    g::Real
    id::Integer
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

    g = v[finite_hl]⋅hl[finite_hl] + w[finite_hu]⋅hu[finite_hu]

    cols = zeros(length(subprob.masterTerms))
    vals = zeros(length(subprob.masterTerms))
    for (s,(i,j,coeff)) in enumerate(subprob.masterTerms)
        cols[s] = j
        vals[s] = -λ[i]*coeff
    end
    G = sparsevec(cols,vals,subprob.nMasterCols)

    return FeasibilityCut(G, g, subprob.id)
end

function (cut::FeasibilityCut)(x::AbstractVector)
    if length(cut.G) != length(x)
        throw(ArgumentError(@sprintf("Dimensions of the cut (%d)) and the given optimization vector (%d) does not match",length(cut.D),length(x))))
    end
    return cut.G⋅x,cut.g
end

function addCut!(lshaped::AbstractLShapedSolver,cut::FeasibilityCut)
    m = lshaped.masterModel
    cutIdx = m.numCols-nscenarios(lshaped)

    D = cut.G
    d = cut.g

    # Scale to avoid numerical issues
    scaling = abs(d)
    if scaling == 0
        scaling = maximum(D)
    end

    D = D/scaling

    # Add feasibility cut
    @constraint(m,sum(D[i]*Variable(m,i)
                      for i = 1:cutIdx) >= sign(d))
    lshaped.nFeasibilityCuts += 1
    if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
        println("Added Feasibility Cut")
    else
        println("Added Feasibility Cut: ", lshaped.masterModel.linconstr[end])
    end
    if istrait(IsRegularized{typeof(lshaped)})
        push!(lshaped.committee,cut)
        addconstr!(lshaped.internal,lowlevel(cut)...)
    end
    return true
end

struct ImproperCut <: AbstractHyperplane
    id::Integer
end
ImproperCut(subprob::SubProblem) = ImproperCut(subprob.id)
proper(cut::ImproperCut) = false
