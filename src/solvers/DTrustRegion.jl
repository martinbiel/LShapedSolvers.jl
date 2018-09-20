@with_kw mutable struct DTrustRegionData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    Δ::T = 1.0
    cΔ::Int = 0
    timestamp::Int = 1
    incubent::Int = 1
    iterations::Int = 0
    major_iterations::Int = 0
    minor_iterations::Int = 0
end

@with_kw mutable struct DTrustRegionParameters{T <: Real}
    κ::T = 0.6
    τ::T = 1e-6
    γ::T = 1e-4
    Δ = 1.0
    Δ̅::T = 1000.0
    bundle::Int = 1
    log::Bool = true
    checkfeas::Bool = false
    autotune::Bool = false
end

"""
    DTrustRegion

Functor object for the distributed trust-region L-shaped algorithm. Create by supplying `:tr` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `κ::Real = 0.6`: Amount of cutting planes, relative to the total number of scenarios, required to generate a new iterate in master procedure.
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `γ::T` = 1e-4: Relative tolerance for deciding if a minor iterate should be accepted as a new major iterate.
- `Δ::Real = 1.0`: Initial size of ∞-norm trust-region.
- `Δ̅::Real = 1000.0`: Maximum size of ∞-norm trust-region.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
- `autotune::Bool = false`: If `true`, heuristic methods are used to set `Δ̅` and `Δ̅` based on the initial decision.
...
"""
struct DTrustRegion{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::DTrustRegionData{T}

    # Master
    mastersolver::M
    mastervector::A
    c::A
    x::A
    Q_history::A

    # Subproblems
    nscenarios::Int
    subobjectives::Vector{A}
    finished::Vector{Int}

    # Workers
    subworkers::Vector{SubWorker{T,A,S}}
    work::Vector{Work}
    decisions::Decisions{A}
    cutqueue::CutQueue{T}
    active_workers::Vector{Future}

    # Trust region
    ξ::A
    incubents::Vector{Int}
    Q̃_history::A
    Δ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::DTrustRegionParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DTrustRegion HasTrustRegion
    @implement_trait DTrustRegion IsParallel

    function (::Type{DTrustRegion})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return TrustRegion(model,ξ₀,mastersolver,subsolver; kw...)
        end
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        mastervector = convert(AbstractVector{T},copy(ξ₀))
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(x₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               DTrustRegionData{T}(),
                               msolver,
                               mastervector,
                               c_,
                               x₀_,
                               A(),
                               n,
                               Vector{A}(),
                               Vector{Int}(),
                               Vector{SubWorker{T,A,S}}(nworkers()),
                               Vector{Work}(nworkers()),
                               RemoteChannel(() -> DecisionChannel(Dict{Int,A}())),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               Vector{Future}(nworkers()),
                               ξ₀_,
                               Vector{Inf}(),
                               A(),
                               A(),
                               A(),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               DTrustRegionParameters{T}(;kw...),
                               ProgressThresh(1.0, "Distributed TR L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
DTrustRegion(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = DTrustRegion(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::DTrustRegion)()
    # Reset timer
    lshaped.progress.tfirst = lshaped.progress.tlast = time()
    # Start workers
    init_workers!(lshaped)
    # Start procedure
    while true
        status = iterate!(lshaped)
        if status != :Valid
            close_workers!(lshaped)
            return status
        end
    end
end

@implement_traitfn function log_regularization!(lshaped::DTrustRegion,HasTrustRegion)
    @unpack Q̃,Δ,incubent = lshaped.solverdata
    push!(lshaped.Q̃_history,Q̃)
    push!(lshaped.Δ_history,Δ)
    push!(lshaped.incubents,incubent)
end

@implement_traitfn function log_regularization!(lshaped::DTrustRegion,t::Integer,HasTrustRegion)
    @unpack Q̃,Δ,incubent = lshaped.solverdata
    lshaped.Q̃_history[t] = Q̃
    lshaped.Δ_history[t] = Δ
    lshaped.incubents[t] = incubent
end

@implement_traitfn function enlarge_trustregion!(lshaped::DTrustRegion,HasTrustRegion)
    @unpack Q,θ = lshaped.solverdata
    @unpack τ,Δ̅ = lshaped.parameters
    t = lshaped.solverdata.timestamp
    lshaped.solverdata.incubent = t
    Δ̃ = lshaped.Δ_history[t]
    ξ = t > 1 ? fetch(lshaped.decisions,lshaped.incubents[t]) : lshaped.ξ
    Q̃ = t > 1 ? lshaped.Q̃_history[lshaped.incubents[t]] : lshaped.solverdata.Q̃
    if Q̃ - Q >= 0.5*(Q̃-θ) && abs(norm(ξ-lshaped.x,Inf) - Δ̃) <= τ
        # Enlarge the trust-region radius
        lshaped.solverdata.Δ = max(lshaped.solverdata.Δ,min(Δ̅,2*Δ̃))
        return true
    else
        return false
    end
end

@implement_traitfn function reduce_trustregion!(lshaped::DTrustRegion,HasTrustRegion)
    @unpack Q,θ = lshaped.solverdata
    t = lshaped.solverdata.timestamp
    Δ̃ = lshaped.Δ_history[t]
    Q̃ = t > 1 ? lshaped.Q̃_history[lshaped.incubents[t]] : lshaped.solverdata.Q̃
    ρ = min(1,Δ̃)*(Q-Q̃)/(Q̃-θ)
    if ρ > 0
        lshaped.solverdata.cΔ += 1
    end
    if ρ > 3 || (lshaped.solverdata.cΔ >= 3 && 1 < ρ <= 3)
        # Reduce the trust-region radius
        lshaped.solverdata.cΔ = 0
        lshaped.solverdata.Δ = min(lshaped.solverdata.Δ,(1/min(ρ,4))*Δ̃)
        return true
    else
        return false
    end
end
