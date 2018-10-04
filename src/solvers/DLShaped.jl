@with_kw mutable struct DLShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
    iterations::Int = 0
end

@with_kw mutable struct DLShapedParameters{T <: Real}
    κ::T = 0.6
    τ::T = 1e-6
    bundle::Int = 1
    log::Bool = true
    checkfeas::Bool = false
end

"""
    DLShaped

Functor object for the distributed L-shaped algorithm. Create by supplying `:dls` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `κ::Real = 0.6`: Amount of cutting planes, relative to the total number of scenarios, required to generate a new iterate in master procedure.
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
...
"""
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
    active_workers::Vector{Future}

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::DLShapedParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DLShaped IsParallel

    function (::Type{DLShaped})(model::JuMP.Model,x₀::AbstractVector,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
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
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
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
                               Vector{SubWorker{T,A,S}}(undef,nworkers()),
                               Vector{Work}(undef,nworkers()),
                               RemoteChannel(() -> DecisionChannel(Dict{Int,A}())),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               Vector{Future}(undef,nworkers()),
                               A(),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               DLShapedParameters{T}(;kw...),
                               ProgressThresh(1.0, "Distributed L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
DLShaped(model::JuMP.Model,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver; kw...) = DLShaped(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::DLShaped)()
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
