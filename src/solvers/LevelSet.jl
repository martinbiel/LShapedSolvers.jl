@with_kw mutable struct LevelSetData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    i::Int = 1
end

@with_kw struct LevelSetParameters{T <: Real}
    τ::T = 1e-6
    λ::T = 1.0
end

struct LevelSet{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::LevelSetData{T}

    # Master
    mastersolver::M
    projectionsolver::M
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
    levels::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::LevelSetParameters{T}

    @implement_trait LevelSet HasLevels

    function (::Type{LevelSet})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(ξ₀_)

        msolver = LQSolver(model,mastersolver)
        psolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               LevelSetData{T}(),
                               msolver,
                               psolver,
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
                               LevelSetParameters{T}(;kw...))
        init!(lshaped,subsolver)

        return lshaped
    end
end
LevelSet(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = LevelSet(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::LevelSet)()
    println("Starting L-Shaped procedure with level sets")
    println("======================")

    println("Main loop")
    println("======================")

    while true
        iterate!(lshaped)

        if check_optimality(lshaped)
            # Optimal
            update_structuredmodel!(lshaped)
            println("Optimal!")
            println("Objective value: ", calculate_objective_value(lshaped))
            println("======================")
            break
        end

        project!(lshaped)
        lshaped.ξ[:] = lshaped.x[:]
    end
end

function iterate!(lshaped::LevelSet)
    if isempty(lshaped.violating)
        # Resolve all subproblems at the current optimal solution
        lshaped.solverdata.Q = resolve_subproblems!(lshaped)
        # Update the optimization vector
        take_step!(lshaped)
    else
        # # Add at most L violating constraints
        # L = 0
        # while !isempty(lshaped.violating) && L < lshaped.nscenarios
        #     constraint = dequeue!(lshaped.violating)
        #     if satisfied(lshaped,constraint)
        #         push!(lshaped.inactive,constraint)
        #         continue
        #     end
        #     println("Adding violated constraint to committee")
        #     push!(lshaped.committee,constraint)
        #     addconstr!(lshaped.mastersolver.lqmodel,lowlevel(constraint)...)
        #     L += 1
        # end
    end

    # Resolve master
    println("Solving master problem")
    lshaped.mastersolver(lshaped.x)
    if status(lshaped.mastersolver) == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    update_solution!(lshaped)
    lshaped.solverdata.θ = calculate_estimate(lshaped)
    # remove_inactive!(lshaped)
    # if length(lshaped.violating) <= lshaped.nscenarios
    #     queueViolated!(lshaped)
    # end
    push!(lshaped.Q_history,lshaped.solverdata.Q)
    push!(lshaped.Q̃_history,lshaped.solverdata.Q̃)
    push!(lshaped.θ_history,lshaped.solverdata.θ)
    nothing
end
