@with_kw mutable struct TrustRegionData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    Δ::T = 1.0
    cΔ::Int = 0
    iterations::Int = 0
    major_iterations::Int = 0
    minor_iterations::Int = 0
end

@with_kw mutable struct TrustRegionParameters{T <: Real}
    τ::T = 1e-6
    γ::T = 1e-4
    Δ::T = 1.0
    Δ̅::T = 1000.0
    bundle::Int = 1
    log::Bool = true
    autotune::Bool = false
end

"""
    TrustRegion

Functor object for the trust-region L-shaped algorithm. Create by supplying `:tr` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `γ::T` = 1e-4: Relative tolerance for deciding if a minor iterate should be accepted as a new major iterate.
- `Δ::Real = 1.0`: Initial size of ∞-norm trust-region.
- `Δ̅::Real = 1000.0`: Maximum size of ∞-norm trust-region.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
- `autotune::Bool = false`: If `true`, heuristic methods are used to set `Δ̅` and `Δ̅` based on the initial decision.
...
"""
struct TrustRegion{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::TrustRegionData{T}

    # Master
    mastersolver::M
    mastervector::A
    c::A
    x::A

    committee::Vector{SparseHyperPlane{T}}
    inactive::Vector{SparseHyperPlane{T}}
    violating::PriorityQueue{SparseHyperPlane{T},T}

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}
    subobjectives::A

    # Trust region
    ξ::A
    Q_history::A
    Q̃_history::A
    Δ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::TrustRegionParameters{T}
    progress::ProgressThresh{T}

    @implement_trait TrustRegion HasTrustRegion

    function (::Type{TrustRegion})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() > 1
            warn("There are worker processes, consider using distributed version of algorithm")
        end
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        mastervector = convert(AbstractVector{T},copy(ξ₀))
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(ξ₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               TrustRegionData{T}(),
                               msolver,
                               mastervector,
                               c_,
                               x₀_,
                               convert(Vector{SparseHyperPlane{T}},linearconstraints(model)),
                               Vector{SparseHyperPlane{T}}(),
                               PriorityQueue{SparseHyperPlane{T},T}(Reverse),
                               n,
                               Vector{SubProblem{T,A,S}}(),
                               A(),
                               ξ₀_,
                               A(),
                               A(),
                               A(),
                               A(),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               TrustRegionParameters{T}(;kw...),
                               ProgressThresh(1.0, "TR L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
TrustRegion(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = TrustRegion(model,rand(model.numCols),mastersolver,subsolver;kw...)

function (lshaped::TrustRegion)()
    # Reset timer
    lshaped.progress.tfirst = lshaped.progress.tlast = time()
    # Start procedure
    while true
        status = iterate!(lshaped)
        if status != :Valid
            return status
        end
    end
end
