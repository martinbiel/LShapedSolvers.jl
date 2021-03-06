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
struct DLevelSet{F, T <: Real, A <: AbstractVector, SP <: StochasticProgram, M <: LQSolver, P <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{F,T,A,M,S}
    stochasticprogram::SP
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
    subworkers::Vector{SubWorker{F,T,A,S}}
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

    @implement_trait DLevelSet LV
    @implement_trait DLevelSet Parallel

    function (::Type{DLevelSet})(stochasticprogram::StochasticProgram, ξ₀::AbstractVector, mastersolver::MPB.AbstractMathProgSolver, subsolver::SubSolver, projectionsolver::MPB.AbstractMathProgSolver, F::Bool; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return LevelSet(stochasticprogram, ξ₀, mastersolver, get_solver(subsolver); kw...)
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
        psolver = LQSolver(first_stage, projectionsolver)
        M = typeof(msolver)
        P = typeof(psolver)
        solver_instance = get_solver(subsolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(solver_instance)),typeof(solver_instance)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        lshaped = new{F,T,A,SP,M,P,S}(stochasticprogram,
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
                                      DLevelSetParameters{T}(;kw...),
                                      ProgressThresh(1.0, "Distributed Leveled L-Shaped Gap "))
        # Initialize solver
        init!(lshaped, subsolver)
        return lshaped
    end
end
DLevelSet(stochasticprogram::StochasticProgram, mastersolver::MPB.AbstractMathProgSolver, subsolver::SubSolver, projectionsolver::MPB.AbstractMathProgSolver, checkfeas::Bool; kw...) = DLevelSet(stochasticprogram, rand(decision_length(stochasticprogram)), mastersolver, subsolver, projectionsolver, checkfeas; kw...)

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
