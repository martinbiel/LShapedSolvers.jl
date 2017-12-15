@with_kw mutable struct RegularizedSolverData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    σ::T = 1.0
    exact_steps::Int = 0
    approximate_steps::Int = 0
    null_steps::Int = 0
end

struct RegularizedLShapedSolver{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMPModel
    solverdata::RegularizedSolverData{T}

    # Master
    mastersolver::M
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
    step_hist::Vector{Int}

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    σ::T
    γ::T
    τ::T

    function (::Type{RegularizedLShapedSolver})(model::JuMPModel,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver)
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:Stochastic) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(ξ₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = num_scenarios(model)

        lshaped = new{T,A,M,S}(model,
                               RegularizedSolverData{T}(),
                               msolver,
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
                               Vector{Int}(),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               convert(T,5.0),
                               convert(T,0.9),
                               convert(T,1e-6)
                               )
        init!(lshaped,subsolver)

        return lshaped
    end
end
RegularizedLShapedSolver(model::JuMPModel,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver) = RegularizedLShapedSolver(model,rand(model.numCols),mastersolver,subsolver)

@implement_trait RegularizedLShapedSolver UsesLocalization IsRegularized

function Base.show(io::IO, lshaped::RegularizedLShapedSolver)
    print(io,"RegularizedLShapedSolver")
end

function (lshaped::RegularizedLShapedSolver)()
    println("Starting L-Shaped procedure with regularized decomposition")
    println("======================")

    println("Main loop")
    println("======================")

    while true
        iterate!(lshaped)

        if check_optimality(lshaped)
            if lshaped.solverdata.Q̃ + lshaped.τ <= -1.2014137491535408e7
                # Optimal
                update_structuredmodel!(lshaped)
                println("Optimal!")
                println("Objective value: ", calculate_objective_value(lshaped))
                println("======================")
                break
            end
        end
    end
end

function iterate!(lshaped::RegularizedLShapedSolver)
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
    # Update master solution
    update_solution!(lshaped)
    lshaped.solverdata.θ = calculate_estimate(lshaped)
    # remove_inactive!(lshaped)
    # if length(lshaped.violating) <= lshaped.nscenarios
    #     queueViolated!(lshaped)
    # end
    push!(lshaped.Q_history,lshaped.solverdata.Q)
    push!(lshaped.Q̃_history,lshaped.solverdata.Q̃)
    push!(lshaped.θ_history,lshaped.solverdata.θ)
    push!(lshaped.σ_history,lshaped.solverdata.σ)
    nothing
end


## Trait functions
# ------------------------------------------------------------
@define_traitfn IsRegularized update_objective!(lshaped::AbstractLShapedSolver)

@implement_traitfn IsRegularized function init_solver!(lshaped::AbstractLShapedSolver)
    lshaped.solverdata.σ = lshaped.σ
    lshaped.solverdata.exact_steps = 0
    lshaped.solverdata.approximate_steps = 0
    lshaped.solverdata.null_steps = 0

    update_objective!(lshaped)
end

@implement_traitfn IsRegularized function take_step!(lshaped::AbstractLShapedSolver)
    Q = lshaped.solverdata.Q
    Q̃ = lshaped.solverdata.Q̃
    θ = lshaped.solverdata.θ
    if abs(θ-Q) <= lshaped.τ*(1+abs(θ))
        println("Exact serious step")
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        lshaped.solverdata.exact_steps += 1
        lshaped.solverdata.σ *= 4
        update_objective!(lshaped)
        push!(lshaped.step_hist,3)
    elseif Q + lshaped.τ*(1+abs(Q)) <= lshaped.γ*Q̃ + (1-lshaped.γ)*θ
        println("Approximate serious step")
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        lshaped.solverdata.approximate_steps += 1
        push!(lshaped.step_hist,2)
    else
        println("Null step")
        lshaped.solverdata.null_steps += 1
        lshaped.solverdata.σ *= 0.9
        update_objective!(lshaped)
        push!(lshaped.step_hist,1)
    end
    nothing
end

@implement_traitfn IsRegularized function update_objective!(lshaped::AbstractLShapedSolver)
    # Linear regularizer penalty
    c = copy(lshaped.c)
    c -= (1/lshaped.solverdata.σ)*lshaped.ξ
    append!(c,fill(1.0,lshaped.nscenarios))
    setobj!(lshaped.mastersolver.lqmodel,c)

    # Quadratic regularizer penalty
    qidx = collect(1:length(lshaped.ξ)+lshaped.nscenarios)
    qval = fill(1/lshaped.solverdata.σ,length(lshaped.ξ))
    append!(qval,zeros(lshaped.nscenarios))
    if applicable(setquadobj!,lshaped.mastersolver.lqmodel,qidx,qidx,qval)
        setquadobj!(lshaped.mastersolver.lqmodel,qidx,qidx,qval)
    else
        error("The regularized decomposition algorithm requires a solver that handles quadratic objectives")
    end
end
