@with_kw mutable struct LShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    iterations::Int = 0
end

@with_kw mutable struct LShapedParameters{T <: Real}
    τ::T = 1e-6
    bundle::Int = 1
    log::Bool = true
end

"""
    LShaped

Functor object for the L-shaped algorithm. Create by supplying `:ls` to the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `bundle::Int = 1`: Amount of cutting planes in bundle. A value of 1 corresponds to a multicut algorithm and a value of at least the number of scenarios yields the classical L-shaped algorithm.
- `log::Bool = true`: Specifices if L-shaped procedure should be logged on standard output or not.
...
"""
struct LShaped{F, T <: Real, A <: AbstractVector, SP <: StochasticProgram, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{F,T,A,M,S}
    stochasticprogram::SP
    solverdata::LShapedData{T}

    # Master
    mastersolver::M
    mastervector::A
    c::A
    x::A
    Q_history::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{F,T,A,S}}
    subobjectives::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::LShapedParameters{T}
    progress::ProgressThresh{T}

    function (::Type{LShaped})(stochasticprogram::StochasticProgram, x₀::AbstractVector, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, F::Bool; kw...)
        if nworkers() > 1
            @warn "There are worker processes, consider using distributed version of algorithm"
        end
        first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
        length(x₀) != first_stage.numCols && error("Incorrect length of starting guess, has ", length(x₀), " should be ", first_stage.numCols)

        T = promote_type(eltype(x₀), Float32)
        c_ = convert(AbstractVector{T}, JuMP.prepAffObjective(first_stage))
        c_ *= first_stage.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T}, copy(x₀))
        mastervector = convert(AbstractVector{T}, copy(x₀))
        A = typeof(x₀_)
        SP = typeof(stochasticprogram)
        msolver = LQSolver(first_stage, mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        lshaped = new{F,T,A,SP,M,S}(stochasticprogram,
                                    LShapedData{T}(),
                                    msolver,
                                    mastervector,
                                    c_,
                                    x₀_,
                                    A(),
                                    n,
                                    Vector{SubProblem{F,T,A,S}}(),
                                    A(),
                                    A(),
                                    Vector{SparseHyperPlane{T}}(),
                                    A(),
                                    LShapedParameters{T}(;kw...),
                                    ProgressThresh(1.0, "L-Shaped Gap "))
        # Initialize solver
        init!(lshaped, subsolver)
        return lshaped
    end
end
LShaped(stochasticprogram::StochasticProgram, mastersolver::MPB.AbstractMathProgSolver, subsolver::MPB.AbstractMathProgSolver, checkfeas::Bool; kw...) = LShaped(stochasticprogram, rand(decision_length(stochasticprogram)), mastersolver, subsolver, checkfeas; kw...)

function (lshaped::LShaped)()
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
