@with_kw mutable struct DLevelSetData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
    iterations::Int = 0
    levelindex::Int = -1
    regularizerindex::Int = -1
end

@with_kw mutable struct DLevelSetParameters{T <: Real}
    κ::T = 0.6
    τ::T = 1e-6
    λ::T = 0.5
    bundle::Int = 1
    log::Bool = true
    checkfeas::Bool = false
    linearize::Bool = false
end

"""
    LevelSet

Functor object for the distributed level-set L-shaped algorithm. Create by supplying `:tr` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `κ::Real = 0.6`: Amount of cutting planes, relative to the total number of scenarios, required to generate a new iterate in master procedure.
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `λ::Real = 0.5`: Controls the level position L = (1-λ)*θ + λ*Q̃, a convex combination of the current lower and upper bound.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
- `linearize::Bool = false`: If `true`, the quadratic terms in the master problem objective are linearized through a ∞-norm approximation.
...
"""
struct DLevelSet{T <: Real, A <: AbstractVector, M <: LQSolver, P <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::DLevelSetData{T}

    # Master
    mastersolver::M
    projectionsolver::P
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

    # Regularizer
    ξ::A
    Q̃_history::A
    levels::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::DLevelSetParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DLevelSet HasLevels
    @implement_trait DLevelSet IsParallel

    function (::Type{DLevelSet})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver,projectionsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return LevelSet(model,ξ₀,mastersolver,subsolver; kw...)
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
        psolver = LQSolver(model,projectionsolver)
        M = typeof(msolver)
        P = typeof(psolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,P,S}(model,
                                 DLevelSetData{T}(),
                                 msolver,
                                 psolver,
                                 mastervector,
                                 c_,
                                 x₀_,
                                 A(),
                                 n,
                                 Vector{A}(),
                                 Vector{Int}(),
                                 Vector{SubWorker{T,A,S}}(undef,nworkers()),
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
                                 DLevelSetParameters{T}(;kw...),
                                 ProgressThresh(1.0, "Distributed Leveled L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
DLevelSet(model::JuMP.Model,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver,projectionsolver::MPB.AbstractMathProgSolver; kw...) = DLevelSet(model,rand(model.numCols),mastersolver,subsolver,projectionsolver; kw...)

function (lshaped::DLevelSet)()
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
