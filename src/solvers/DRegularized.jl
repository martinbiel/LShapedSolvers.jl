@with_kw mutable struct DRegularizedData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    σ::T = 1.0
    timestamp::Int = 1
    iterations::Int = 0
    major_iterations::Int = 0
    minor_iterations::Int = 0
    regularizerindex::Int = -1
end

@with_kw mutable struct DRegularizedParameters{T <: Real}
    κ::T = 0.6
    τ::T = 1e-6
    γ::T = 0.9
    σ::T = 1.0
    σ̅::T = 4.0
    σ̲::T = 0.5
    bundle::Int = 1
    log::Bool = true
    autotune::Bool = false
    linearize::Bool = false
end

"""
    DRegularized

Functor object for the distributed regularized L-shaped algorithm. Create by supplying `:drd` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `κ::Real = 0.6`: Amount of cutting planes, relative to the total number of scenarios, required to generate a new iterate in master procedure.
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `σ::Real = 1.0`: Initial value of regularization parameter. Controls the relative penalty of the deviation from the current major iterate.
- `σ̅::real = 4.0`: Maximum value of the regularization parameter.
- `σ̲::real = 0.5`: Minimum value of the regularization parameter.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
- `autotune::Bool = false`: If `true`, heuristic methods are used to set `σ, σ̅` and `σ̲` based on the initial decision.
- `linearize::Bool = false`: If `true`, the quadratic terms in the master problem objective are linearized through a ∞-norm approximation.
...
"""
struct DRegularized{F, T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{F,T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::DRegularizedData{T}

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
    subworkers::Vector{SubWorker{F,T,A,S}}
    work::Vector{Work}
    decisions::Decisions{A}
    cutqueue::CutQueue{T}
    active_workers::Vector{Future}

    # Trust region
    ξ::A
    Q̃_history::A
    σ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::DRegularizedParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DRegularized IsRegularized
    @implement_trait DRegularized IsParallel

    function (::Type{DRegularized})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver,F::Bool; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return Regularized(model,ξ₀,mastersolver,subsolver; kw...)
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
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{F,T,A,M,S}(model,
                                 DRegularizedData{T}(),
                                 msolver,
                                 mastervector,
                                 c_,
                                 x₀_,
                                 A(),
                                 n,
                                 Vector{A}(),
                                 Vector{Int}(),
                                 Vector{SubWorker{F,T,A,S}}(undef,nworkers()),
                                 Vector{Work}(undef,nworkers()),
                                 RemoteChannel(() -> DecisionChannel(Dict{Int,A}())),
                                 RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                                 Vector{Future}(undef,nworkers()),
                                 ξ₀_,
                                 A(),
                                 A(),
                                 A(),
                                 Vector{SparseHyperPlane{T}}(),
                                 A(),
                                 DRegularizedParameters{T}(;kw...),
                                 ProgressThresh(1.0, "Distributed RD L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
DRegularized(model::JuMP.Model,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver,checkfeas::Bool; kw...) = DRegularized(model,rand(model.numCols),mastersolver,subsolver,checkfeas; kw...)

function (lshaped::DRegularized)()
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
