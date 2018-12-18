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
struct LevelSet{F, T <: Real, A <: AbstractVector, SP <: StochasticProgram, M <: LQSolver, P <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{F,T,A,M,S}
    stochasticprogram::SP
    solverdata::LevelSetData{T}

    # Master
    mastersolver::M
    projectionsolver::P
    mastervector::A
    c::A
    x::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{F,T,A,S}}
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

    @implement_trait LevelSet LV

    function (::Type{LevelSet})(stochasticprogram::StochasticProgram, ξ₀::AbstractVector, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, projectionsolver::MPB.AbstractMathProgSolver, F::Bool; kw...)
        if nworkers() > 1
            @warn "There are worker processes, consider using distributed version of algorithm"
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
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        lshaped = new{F,T,A,SP,M,P,S}(stochasticprogram,
                                      LevelSetData{T}(),
                                      msolver,
                                      psolver,
                                      mastervector,
                                      c_,
                                      x₀_,
                                      n,
                                      Vector{SubProblem{F,T,A,S}}(),
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
        init!(lshaped, subsolver)
        return lshaped
    end
end
LevelSet(stochasticprogram::StochasticProgram, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, projectionsolver::MPB.AbstractMathProgSolver, checkfeas::Bool; kw...) = LevelSet(stochasticprogram, rand(decision_length(stochasticprogram)), mastersolver, subsolver, projectionsolver, checkfeas; kw...)

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
