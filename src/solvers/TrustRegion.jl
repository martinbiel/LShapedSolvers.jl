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
    checkfeas::Bool = false
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
struct TrustRegion{F, T <: Real, A <: AbstractVector, SP <: StochasticProgram, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{F,T,A,M,S}
    stochasticprogram::SP
    solverdata::TrustRegionData{T}

    # Master
    mastersolver::M
    mastervector::A
    c::A
    x::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{F,T,A,S}}
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

    function (::Type{TrustRegion})(stochasticprogram::StochasticProgram, ξ₀::AbstractVector, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, F::Bool; kw...)
        if nworkers() > 1
            @warn "There are worker processes, consider using distributed version of algorithm"
        end
        first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
        length(ξ₀) != first_stage.numCols && error("Incorrect length of starting guess, has ", length(ξ₀), " should be ", first_stage.numCols)

        T = promote_type(eltype(ξ₀), Float32)
        c_ = convert(AbstractVector{T}, JuMP.prepAffObjective(first_stage))
        c_ *= first_stage.objSense == :Min ? 1 : -1
        ξ₀_ = convert(AbstractVector{T}, copy(ξ₀))
        x₀_ = convert(AbstractVector{T}, copy(ξ₀))
        mastervector = convert(AbstractVector{T}, copy(ξ₀))
        A = typeof(ξ₀_)
        SP = typeof(stochasticprogram)
        msolver = LQSolver(first_stage, mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        lshaped = new{F,T,A,SP,M,S}(stochasticprogram,
                                    TrustRegionData{T}(),
                                    msolver,
                                    mastervector,
                                    c_,
                                    x₀_,
                                    n,
                                    Vector{SubProblem{F,T,A,S}}(),
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
        init!(lshaped, subsolver)
        return lshaped
    end
end
TrustRegion(stochasticprogram::StochasticProgram, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, checkfeas::Bool; kw...) = TrustRegion(stochasticprogram, rand(decision_length(stochasticprogram)), mastersolver, subsolver, checkfeas; kw...)

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
