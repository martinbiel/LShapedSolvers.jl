abstract type HyperPlaneType end

abstract type OptimalityCut <: HyperPlaneType end
abstract type FeasibilityCut <: HyperPlaneType end
abstract type LinearConstraint <: HyperPlaneType end
abstract type Infeasible <: HyperPlaneType end
abstract type Unbounded <: HyperPlaneType end

struct HyperPlane{H <: HyperPlaneType, T <: Real, A <: AbstractVector}
    δQ::A
    q::T
    id::Int

    function (::Type{HyperPlane})(δQ::AbstractVector, q::Real, id::Int, ::Type{H}) where H <: HyperPlaneType
        T = promote_type(eltype(δQ), Float32)
        δQ_ = convert(AbstractVector{T}, δQ)
        new{H, T, typeof(δQ_)}(δQ_, q, id)
    end
end
OptimalityCut(δQ::AbstractVector, q::Real, id::Int) = HyperPlane(δQ, q, id, OptimalityCut)
FeasibilityCut(δQ::AbstractVector, q::Real, id::Int) = HyperPlane(δQ, q, id, FeasibilityCut)
LinearConstraint(δQ::AbstractVector, q::Real, id::Int) = HyperPlane(δQ, q, id, LinearConstraint)
Infeasible(id::Int) = HyperPlane(sparsevec(Float64[]), 1e10, id, Infeasible)
Unbounded(id::Int) = HyperPlane(sparsevec(Float64[]), 1e10, id, Unbounded)

SparseHyperPlane{T <: Real} = HyperPlane{<:HyperPlaneType, T, SparseVector{T,Int64}}
SparseOptimalityCut{T <: Real} = HyperPlane{OptimalityCut, T, SparseVector{T,Int64}}
SparseFeasibilityCut{T <: Real} = HyperPlane{FeasibilityCut, T, SparseVector{T,Int64}}
SparseLinearConstraint{T <: Real} = HyperPlane{LinearConstraint, T, SparseVector{T,Int64}}

function (hyperplane::HyperPlane{FeasibilityCut})(x::AbstractVector)
    return Inf
end
function (cut::HyperPlane{OptimalityCut})(x::AbstractVector)
    if length(cut.δQ) != length(x)
        throw(ArgumentError(@sprintf("Dimensions of the cut (%d)) and the given optimization vector (%d) does not match", length(cut.δQ), length(x))))
    end
    return cut.q-cut.δQ⋅x
end
function (hyperplane::HyperPlane{Infeasible})(x::AbstractVector)
    return Inf
end
function (hyperplane::HyperPlane{Unbounded})(x::AbstractVector)
    return -Inf
end

infeasible(hyperplane::HyperPlane) = true
infeasible(hyperplane::HyperPlane{Infeasible}) = true
bounded(hyperplane::HyperPlane) = true
bounded(hyperplane::HyperPlane{Unbounded}) = false
function optimal(cut::HyperPlane{OptimalityCut},x::AbstractVector, θ::Real, τ::Real)
    Q = cut(x)
    return θ > -Inf && abs(θ-Q) <= τ*(1+abs(Q))
end
function active(hyperplane::HyperPlane, x::AbstractVector, τ::Real)
    return abs(gap(hyperplane,x)) <= τ
end
function satisfied(hyperplane::HyperPlane, x::AbstractVector, τ::Real)
    return gap(hyperplane,x) >= -τ
end
function satisfied(cut::HyperPlane{OptimalityCut}, x::AbstractVector, θ::Real, τ::Real)
    Q = cut(x)
    return θ > -Inf && θ >= Q - τ
end
function gap(hyperplane::HyperPlane,x::AbstractVector)
    if length(hyperplane.δQ) != length(x)
        throw(ArgumentError(@sprintf("Dimensions of the cut (%d)) and the given optimization vector (%d) does not match", length(hyperplane.δQ), length(x))))
    end
    return hyperplane.δQ⋅x-hyperplane.q
end
function gap(cut::HyperPlane{OptimalityCut}, x::AbstractVector, θ::Real)
    if θ > -Inf
        return θ-cut(x)
    else
        return Inf
    end
end
function lowlevel(hyperplane::HyperPlane{H,T,SparseVector{T,Int}}) where {H <: HyperPlaneType, T <: Real}
    return hyperplane.δQ.nzind, hyperplane.δQ.nzval, hyperplane.q, Inf
end
function lowlevel(cut::HyperPlane{OptimalityCut,T,SparseVector{T,Int}}) where T <: Real
    nzind = copy(cut.δQ.nzind)
    nzval = copy(cut.δQ.nzval)
    push!(nzind,length(cut.δQ)+cut.id)
    push!(nzval,1.0)
    return nzind, nzval, cut.q, Inf
end

# Constructors #
# ======================================================================== #
function OptimalityCut(subproblem::SubProblem)
    λ = getduals(subproblem.solver)
    π = subproblem.π
    cols = zeros(length(subproblem.masterterms))
    vals = zeros(length(subproblem.masterterms))
    for (s,(i,j,coeff)) in enumerate(subproblem.masterterms)
        cols[s] = j
        vals[s] = -π*λ[i]*coeff
    end
    δQ = sparsevec(cols,vals, length(subproblem.x))
    q = π*getobjval(subproblem.solver)+δQ⋅subproblem.x

    return OptimalityCut(δQ, q, subproblem.id)
end
ArtificialCut(val::Real,dim::Int,id::Int) = OptimalityCut(sparsevec(zeros(dim)), val, id)

function FeasibilityCut(subproblem::SubProblem)
    λ = getduals(subproblem.feasibility_solver)
    cols = zeros(length(subproblem.masterterms))
    vals = zeros(length(subproblem.masterterms))
    for (s, (i,j,coeff)) in enumerate(subproblem.masterterms)
        cols[s] = j
        vals[s] = -λ[i]*coeff
    end
    G = sparsevec(cols,vals,length(subproblem.x))
    g = getobjval(subproblem.feasibility_solver)+G⋅subproblem.x

    return FeasibilityCut(G, g, subproblem.id)
end

function LinearConstraint(constraint::JuMP.LinearConstraint, i::Integer)
    sense = JuMP.sense(constraint)
    if sense == :range
        throw(ArgumentError("Cannot handle range constraints"))
    end
    cols = map(v->v.col, constraint.terms.vars)
    vals = constraint.terms.coeffs * (sense == :(>=) ? 1 : -1)
    G = sparsevec(cols, vals, constraint.terms.vars[1].m.numCols)
    g = JuMP.rhs(constraint) * (sense == :(>=) ? 1 : -1)

    return LinearConstraint(G, g, i)
end

function linearconstraints(m::JuMP.Model)
    constraints = Vector{HyperPlane{LinearConstraint}}(length(m.linconstr))
    for (i, c) in enumerate(m.linconstr)
        constraints[i] = LinearConstraint(c, i)
    end
    return constraints
end

Infeasible(subprob::SubProblem) = Infeasible(subprob.id)
Unbounded(subprob::SubProblem) = Unbounded(subprob.id)

function zero(h::HyperPlane{H,T,A}) where {H <: HyperPlaneType, T <: Real, A <: AbstractVector}
    return HyperPlane(zero(h.δQ), zero(T), h.id, H)
end
function +(h1::HyperPlane{H,T,A},h2::HyperPlane{H,T,A}) where {H <: HyperPlaneType, T <: Real, A <: AbstractVector}
    return HyperPlane(h1.δQ + h2.δQ, h1.q + h2.q, h1.id, H)
end

mutable struct CutBundle{T <: Real}
    cuts::Vector{SparseOptimalityCut{T}}
    q::T

    function (::Type{CutBundle})(::Type{T}) where T <: Real
        new{T}(Vector{SparseOptimalityCut{T}}(), zero(T))
    end
end
length(bundle::CutBundle) = length(bundle.cuts)
function aggregate!(bundle::CutBundle)
    aggregated_cut = zero(bundle.cuts[1])
    for j = 1:length(bundle)
        aggregated_cut += pop!(bundle.cuts)
    end
    return aggregated_cut
end

# ======================================================================== #
