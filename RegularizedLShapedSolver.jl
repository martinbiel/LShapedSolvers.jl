mutable struct RegularizedLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterSolver::AbstractLQSolver
    c::AbstractVector
    x::AbstractVector
    obj::Real

    committee::Vector{AbstractHyperplane}
    inactive::Vector{AbstractHyperplane}
    violating::PriorityQueue

    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Regularizer
    ξ::AbstractVector
    Q̃::Real
    Q̃_hist::AbstractVector
    Δ̅::Real
    Δ̅_hist::AbstractVector
    nExactSteps::Integer
    nApproximateSteps::Integer
    nNullSteps::Integer

    # Cuts
    θs
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer
    cuts::Vector{AbstractHyperplane}

    status::Symbol
    σ::Real
    γ::Real
    τ::Real

    function RegularizedLShapedSolver(m::JuMPModel,ξ::Vector{Float64})
        lshaped = new(m)

        if length(ξ) != m.numCols
            throw(ArgumentError(string("Incorrect length of regularizer, has ",length(ξ)," should be ",m.numCols)))
        end

        lshaped.x = ξ
        lshaped.ξ = ξ
        lshaped.Q̃_hist = Float64[]
        lshaped.σ = 1.0
        lshaped.γ = 0.9
        lshaped.Δ̅ = max(1.0,0.2*norm(ξ,Inf))
        lshaped.Δ̅_hist = [lshaped.Δ̅]
        lshaped.nExactSteps = 0
        lshaped.nApproximateSteps = 0
        lshaped.nNullSteps = 0

        init(lshaped)

        lshaped.committee = linearconstraints(lshaped.structuredModel)
        lshaped.inactive = Vector{AbstractHyperplane}()
        lshaped.violating = PriorityQueue(Reverse)

        return lshaped
    end
end

@traitimpl IsRegularized{RegularizedLShapedSolver}

RegularizedLShapedSolver(m::JuMPModel,ξ::AbstractVector) = RegularizedLShapedSolver(m,convert(Vector{Float64},ξ))

function Base.show(io::IO, lshaped::RegularizedLShapedSolver)
    print(io,"RegularizedLShapedSolver")
end

function prepareMaster!(lshaped::RegularizedLShapedSolver,n::Integer)
    lshaped.masterSolver = LQSolver(lshaped.structuredModel)
    # θs
    for i = 1:n
        addvar!(lshaped.masterSolver.model,-Inf,Inf,1.0)
    end
    lshaped.masterSolver.x = copy(lshaped.x)
    append!(lshaped.masterSolver.x,fill(-Inf,n))

    lshaped.c = getobj(lshaped.masterSolver.model)
    updateObjective!(lshaped)
end

function updateObjective!(lshaped::RegularizedLShapedSolver)
    # Linear regularizer penalty
    c = copy(lshaped.c)
    c -= (1/lshaped.σ)*lshaped.ξ
    append!(c,fill(1.0,lshaped.nscenarios))
    setobj!(lshaped.masterSolver.model,c)

    # Quadratic regularizer penalty
    qidx = collect(1:length(lshaped.ξ)+lshaped.nscenarios)
    qval = fill(1/lshaped.σ,length(lshaped.ξ))
    append!(qval,zeros(lshaped.nscenarios))
    setquadobj!(lshaped.masterSolver.model,qidx,qidx,qval)
end

@traitfn function removeInactive!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    inactive = find(c->!active(lshaped,c),lshaped.committee)
    diff = length(lshaped.committee) - length(lshaped.structuredModel.linconstr) - lshaped.nscenarios
    if isempty(inactive) || diff <= 0
        return false
    end
    if diff <= length(inactive)
        inactive = inactive[1:diff]
    end
    append!(lshaped.inactive,lshaped.committee[inactive])
    deleteat!(lshaped.committee,inactive)
    delconstrs!(lshaped.masterSolver.model,inactive)
    return true
end

@traitfn function takeRegularizedStep!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    θ = sum(lshaped.θs)
    if abs(θ-lshaped.obj) <= lshaped.τ*(1+abs(lshaped.obj))
        println("Exact serious step")
        lshaped.Δ̅ = norm(lshaped.x-lshaped.ξ,Inf)
        push!(lshaped.Δ̅_hist,lshaped.Δ̅)
        lshaped.ξ = copy(lshaped.x)
        lshaped.Q̃ = lshaped.obj
        push!(lshaped.Q̃_hist,lshaped.Q̃)
        lshaped.nExactSteps += 1
        lshaped.σ *= 4
        updateObjective!(lshaped)
    elseif lshaped.obj <= lshaped.γ*lshaped.Q̃ + (1-lshaped.γ)*θ + lshaped.τ
        println("Approximate serious step")
        lshaped.Δ̅ = norm(lshaped.x-lshaped.ξ,Inf)
        push!(lshaped.Δ̅_hist,lshaped.Δ̅)
        lshaped.ξ = copy(lshaped.x)
        lshaped.Q̃ = lshaped.obj
        push!(lshaped.Q̃_hist,lshaped.Q̃)
        lshaped.nApproximateSteps += 1
    else
        println("Null step")
        lshaped.nNullSteps += 1
        lshaped.σ *= 0.9
        updateObjective!(lshaped)
    end
end

function (lshaped::RegularizedLShapedSolver)()
    println("Starting regularized L-Shaped procedure\n")
    println("======================")
    # Initial solve of subproblems at starting guess
    println("Initial solve of subproblems at starting regularizer")
    updateSubProblems!(lshaped.subProblems,lshaped.ξ)
    map(s->addCut!(lshaped,s(),lshaped.ξ),lshaped.subProblems)
    lshaped.Q̃ = sum(lshaped.subObjectives)
    push!(lshaped.Q̃_hist,lshaped.Q̃)
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    lshaped.status = status(lshaped.masterSolver)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateSolution!(lshaped)

    println("Main loop")
    println("======================")

    while true
        if isempty(lshaped.violating)
            # Resolve all subproblems at the current optimal solution
            resolveSubproblems!(lshaped)
            # Update the objective value
            updateObjectiveValue!(lshaped)
            # Update the optimization vector
            takeRegularizedStep!(lshaped)
        else
            # Add at most L violating constraints
            L = 0
            while !isempty(lshaped.violating) && L < lshaped.nscenarios
                constraint = dequeue!(lshaped.violating)
                if satisfied(lshaped,constraint)
                    push!(lshaped.inactive,constraint)
                    continue
                end
                println("Adding violated constraint to committee")
                push!(lshaped.committee,constraint)
                addconstr!(lshaped.masterSolver.model,lowlevel(constraint)...)
                L += 1
            end
        end

        # Resolve master
        println("Solving master problem")
        lshaped.masterSolver()
        lshaped.status = status(lshaped.masterSolver)
        if lshaped.status == :Infeasible
            println("Master is infeasible, aborting procedure.")
            println("======================")
            return
        end
        # if lshaped.status != :Optimal
        #     setparam!(lshaped.gurobienv,"Presolve",2)
        #     setparam!(lshaped.gurobienv,"BarHomogeneous",1)
        #     lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
        #     setsolver(lshaped.masterModel,lshaped.masterSolver)
        #     optimize!(lshaped.internal)
        #     lshaped.status = status(lshaped.internal)
        #     if lshaped.status == :Optimal
        #         setparam!(lshaped.gurobienv,"Presolve",-1)
        #         setparam!(lshaped.gurobienv,"BarHomogeneous",-1)
        #         lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
        #         setsolver(lshaped.masterModel,lshaped.masterSolver)
        #     else
        #         if lshaped.status == :Infeasible
        #             println("Master is infeasible, aborting procedure.")
        #         else
        #             println("Master could not be solved, aborting procedure")
        #         end
        #         println("======================")
        #         return
        #     end
        # end
        # Update master solution
        updateSolution!(lshaped)
        removeInactive!(lshaped)
        if length(lshaped.violating) <= lshaped.nscenarios
            queueViolated!(lshaped)
        end

        if checkOptimality(lshaped)
            if norm(lshaped.x-lshaped.ξ,Inf) - 1.1*lshaped.Δ̅ <= lshaped.τ
                lshaped.σ *= 4
            else
                # Optimal
                lshaped.status = :Optimal
                updateStructuredModel!(lshaped)
                println("Optimal!")
                println("Objective value: ", lshaped.Q̃)
                println("======================")
                break
            end
        end
    end
end
