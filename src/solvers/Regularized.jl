@with_kw mutable struct RegularizedData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    σ::T = 1.0
    iterations::Int = 0
    major_iterations::Int = 0
    minor_iterations::Int = 0
    regularizerindex::Int = -1
end

@with_kw mutable struct RegularizedParameters{T <: Real}
    τ::T = 1e-6
    σ::T = 1.0
    σ̅::T = 4.0
    σ̲::T = 0.5
    bundle::Int = 1
    log::Bool = true
    checkfeas::Bool = false
    autotune::Bool = false
    linearize::Bool = false
end

"""
    Regularized

Functor object for the regularized decomposition L-shaped algorithm. Create by supplying `:rd` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
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
                               A(),
                               ξ₀_,
                               A(),
                               A(),
                               A(),
                               A(),
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
