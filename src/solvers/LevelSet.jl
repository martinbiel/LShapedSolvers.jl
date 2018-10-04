@with_kw mutable struct LevelSetData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    iterations::Int = 0
    levelindex::Int = -1
    regularizerindex::Int = -1
end

@with_kw mutable struct LevelSetParameters{T <: Real}
    τ::T = 1e-6
    λ::T = 0.5
    bundle::Int = 1
    log::Bool = true
    checkfeas::Bool = false
    linearize::Bool = false
end

"""
    LevelSet

Functor object for the level-set L-shaped algorithm. Create by supplying `:tr` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `λ::Real = 0.5`: Controls the level position L = (1-λ)*θ + λ*Q̃, a convex combination of the current lower and upper bound.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
- `linearize::Bool = false`: If `true`, the quadratic terms in the master problem objective are linearized through a ∞-norm approximation.
...
"""
struct LevelSet{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::LevelSetData{T}

    # Master
    mastersolver::M
    projectionsolver::M
    mastervector::A
    c::A
    x::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}
    subobjectives::A

    # Regularizer
    ξ::A
    Q̃_history::A
    Q_history::A
    levels::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::LevelSetParameters{T}
    progress::ProgressThresh{T}

    @implement_trait LevelSet HasLevels

    function (::Type{LevelSet})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() > 1
            @warn "There are worker processes, consider using distributed version of algorithm"
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
        psolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               LevelSetData{T}(),
                               msolver,
                               psolver,
                               mastervector,
                               c_,
                               x₀_,
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
                               LevelSetParameters{T}(;kw...),
                               ProgressThresh(1.0, "Leveled L-Shaped Gap "))
        # Initialize solver
        init!(lshaped,subsolver)
        return lshaped
    end
end
LevelSet(model::JuMP.Model,mastersolver::MPB.AbstractMathProgSolver,subsolver::MPB.AbstractMathProgSolver; kw...) = LevelSet(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::LevelSet)()
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
