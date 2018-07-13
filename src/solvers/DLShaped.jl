@with_kw mutable struct DLShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
    iterations::Int = 0
end

@with_kw mutable struct DLShapedParameters{T <: Real}
    κ::T = 0.3
    τ::T = 1e-5
    log::Bool = true
end

struct DLShaped{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::DLShapedData{T}

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

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::DLShapedParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DLShaped IsParallel

    function (::Type{DLShaped})(model::JuMP.Model,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return LShaped(model,x₀,mastersolver,subsolver; kw...)
        end
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(x₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        mastervector = convert(AbstractVector{T},copy(x₀))
        x₀_ = convert(AbstractVector{T},copy(x₀))
        A = typeof(x₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               DLShapedData{T}(),
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
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               DLShapedParameters{T}(;kw...),
                               ProgressThresh(1.0, "Distributed L-Shaped Gap "))
        lshaped.progress.thresh = lshaped.parameters.τ
        push!(lshaped.subobjectives,zeros(n))
        push!(lshaped.finished,0)
        push!(lshaped.Q_history,Inf)
        push!(lshaped.θ_history,-Inf)

        init!(lshaped,subsolver)

        return lshaped
    end
end
DLShaped(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = DLShaped(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::DLShaped)()
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
