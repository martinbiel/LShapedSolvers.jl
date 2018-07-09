@with_kw mutable struct DLevelSetData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
    iterations::Int = 0
end

@with_kw mutable struct DLevelSetParameters{T <: Real}
    κ::T = 0.3
    τ::T = 1e-6
    λ::T = 0.5
    log::Bool = true
end

struct DLevelSet{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::DLevelSetData{T}

    # Master
    mastersolver::M
    projectionsolver::M
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

    function (::Type{DLevelSet})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
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
        psolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
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
                               Vector{SubWorker{T,A,S}}(nworkers()),
                               Vector{Work}(nworkers()),
                               RemoteChannel(() -> DecisionChannel(Dict{Int,A}())),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               ξ₀_,
                               A(),
                               A(),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               DLevelSetParameters{T}(;kw...),
                               ProgressThresh(1.0, "Distributed Leveled L-Shaped Gap "))
        lshaped.progress.thresh = lshaped.parameters.τ
        push!(lshaped.subobjectives,zeros(n))
        push!(lshaped.finished,0)
        push!(lshaped.Q_history,Inf)
        push!(lshaped.Q̃_history,Inf)
        push!(lshaped.θ_history,-Inf)

        init!(lshaped,subsolver)

        return lshaped
    end
end
DLevelSet(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = DLevelSet(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::DLevelSet)()
    # Reset timer
    lshaped.progress.tfirst = lshaped.progress.tlast = time()
    # Start workers
    active_workers = init_workers!(lshaped)
    # Start procedure
    while true
        status = iterate!(lshaped)
        if status != :Valid
            close_workers!(lshaped,active_workers)
            return status
        end
    end
end
