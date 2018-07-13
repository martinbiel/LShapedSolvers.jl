@with_kw mutable struct RegularizedData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    σ::T = 1.0
    iterations::Int = 0
    major_iterations::Int = 0
    minor_iterations::Int = 0
end

@with_kw mutable struct RegularizedParameters{T <: Real}
    τ::T = 1e-5
    γ::T = 0.9
    σ::T = 1.0
    σ̅::T = 4.0
    σ̲::T = 0.5
    log::Bool = true
    autotune::Bool = false
end

struct Regularized{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::RegularizedData{T}

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

    # Regularizer
    ξ::A
    Q̃_history::A
    Q_history::A
    σ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::RegularizedParameters{T}
    progress::ProgressThresh{T}

    @implement_trait Regularized IsRegularized

    function (::Type{Regularized})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
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
                               RegularizedData{T}(),
                               msolver,
                               mastervector,
                               c_,
                               x₀_,
                               convert(Vector{SparseHyperPlane{T}},linearconstraints(model)),
                               Vector{SparseHyperPlane{T}}(),
                               PriorityQueue{SparseHyperPlane{T},T}(Reverse),
                               n,
                               Vector{SubProblem{T,A,S}}(),
                               A(zeros(n)),
                               ξ₀_,
                               A(),
                               A(),
                               A(),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               RegularizedParameters{T}(;kw...),
                               ProgressThresh(1.0, "RD L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
Regularized(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = Regularized(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::Regularized)()
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
