mutable struct RegularizedLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    internal::AbstractLinearQuadraticModel
    masterSolver::AbstractMathProgSolver
    gurobienv::Gurobi.Env
    x::AbstractVector
    obj::Real

    committee::Vector{AbstractHyperplane}
    inactive::Vector{AbstractHyperplane}
    violating::PriorityQueue

    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Regularizer
    a::AbstractVector
    Qa::Real
    nExactSteps::Integer
    nApproximateSteps::Integer
    nNullSteps::Integer

    # Cuts
    θs
    ready::BitArray
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer
    cuts::Vector{AbstractHyperplane}

    status::Symbol
    σ::Real
    γ::Real
    τ::Real
    p::Real
    ref::Real

    function RegularizedLShapedSolver(m::JuMPModel,a::Vector{Float64})
        lshaped = new(m)

        if length(a) != m.numCols
            throw(ArgumentError(string("Incorrect length of regularizer, has ",length(a)," should be ",m.numCols)))
        end

        lshaped.a = a
        lshaped.σ = 1.0
        lshaped.γ = 0.9
        lshaped.p = 1.0
        lshaped.nExactSteps = 0
        lshaped.nApproximateSteps = 0
        lshaped.nNullSteps = 0

        init(lshaped)

        lshaped.committee = linearconstraints(lshaped.structuredModel)
        lshaped.inactive = Vector{AbstractHyperplane}()
        lshaped.violating = PriorityQueue(Reverse)

        lshaped.Qa = calculateObjective(lshaped,a)

        return lshaped
    end
end

@traitimpl IsRegularized{RegularizedLShapedSolver}

RegularizedLShapedSolver(m::JuMPModel,a::AbstractVector,ref::Real) = RegularizedLShapedSolver(m,convert(Vector{Float64},a))

function Base.show(io::IO, lshaped::RegularizedLShapedSolver)
    print(io,"RegularizedLShapedSolver")
end

@traitfn function prepareMaster!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS,n)
    lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
    lshaped.ready = falses(n)

    updateObjective!(lshaped)

    lshaped.gurobienv = Gurobi.Env()
    setparam!(lshaped.gurobienv,"OutputFlag",0)
    lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
    setsolver(lshaped.masterModel,lshaped.masterSolver)
    JuMP.build(lshaped.masterModel)
    lshaped.internal = copy(lshaped.masterModel.internalModel)
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
    delconstrs!(lshaped.internal,inactive)
    return true
end

@traitfn function queueViolated!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    violating = find(c->violated(lshaped,c),lshaped.inactive)
    if isempty(violating)
        return false
    end
    gaps = map(c->gap(lshaped,c),lshaped.inactive[violating])
    if isempty(lshaped.violating)
        lshaped.violating = PriorityQueue(Reverse,zip(lshaped.inactive[violating],gaps))
    else
        for (c,g) in zip(lshaped.inactive[violating],gaps)
            enqueue!(lshaped.violating,c,g)
        end
    end
    deleteat!(lshaped.inactive,violating)
    return true
end

@traitfn function updateObjective!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    x = [Variable(lshaped.masterModel,i) for i in 1:(lshaped.structuredModel.numCols)]

    @objective(lshaped.masterModel,Min,sum(c.*x) + sum(lshaped.θs[lshaped.ready]) + (1/(2*lshaped.σ))*sum((x-lshaped.a).*(x-lshaped.a)))
end

@traitfn function takeRegularizedStep!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    θ = sum(getvalue(lshaped.θs))
    if abs(θ-lshaped.obj) <= lshaped.τ*(1+abs(lshaped.obj))
        println("Exact serious step")
        lshaped.a = copy(lshaped.x)
        lshaped.Qa = lshaped.obj
        lshaped.nExactSteps += 1
    elseif lshaped.obj <= lshaped.γ*lshaped.Qa + (1-lshaped.γ)*θ + lshaped.τ && approximate(lshaped)
        println("Approximate serious step")
        lshaped.a = copy(lshaped.x)
        lshaped.Qa = lshaped.obj
        lshaped.nApproximateSteps += 1
    else
        println("Null step")
        lshaped.nNullSteps += 1
    end
end

# Regularized implementation
@traitfn function checkOptimality{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    c = JuMP.prepAffObjective(lshaped.structuredModel)

    z = c⋅lshaped.x + sum(getvalue(lshaped.θs[lshaped.ready]))

    @show z
    @show lshaped.Qa
    @show abs(z-lshaped.Qa)

    if abs(z - lshaped.Qa) <= lshaped.τ*(1+abs(lshaped.Qa))
        return true
    else
        return false
    end
end

function approximate(lshaped::RegularizedLShapedSolver)
    nactive = count(map(c->active(lshaped,c),lshaped.committee))
    return nactive == length(lshaped.structuredModel.linconstr) + lshaped.nscenarios
end

function (lshaped::RegularizedLShapedSolver)()
    println("Starting L-Shaped procedure\n")
    println("======================")
    # Initial solve of master problem
    println("Initial solve of master")
    #lshaped.status = solve(lshaped.masterModel)
    optimize!(lshaped.internal)
    lshaped.masterModel.colVal = getsolution(lshaped.internal)[1:lshaped.masterModel.numCols]
    lshaped.status = status(lshaped.internal)
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

            takeRegularizedStep!(lshaped)

            c = MathProgBase.SolverInterface.getobj(lshaped.internal)
            c[end-lshaped.nscenarios+1:end] = lshaped.p
            MathProgBase.SolverInterface.setobj!(lshaped.internal,c)
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
                addconstr!(lshaped.internal,lowlevel(constraint)...)
                L += 1
            end
        end

        # Resolve master
        println("Solving master problem")
        #lshaped.status = solve(lshaped.masterModel)
        optimize!(lshaped.internal)

        lshaped.status = status(lshaped.internal)
        if lshaped.status != :Optimal
            setparam!(lshaped.gurobienv,"Presolve",2)
            setparam!(lshaped.gurobienv,"BarHomogeneous",1)
            lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
            setsolver(lshaped.masterModel,lshaped.masterSolver)
            #lshaped.status = solve(lshaped.masterModel)
            optimize!(lshaped.internal)
            lshaped.status = status(lshaped.internal)
            if lshaped.status == :Optimal
                setparam!(lshaped.gurobienv,"Presolve",-1)
                setparam!(lshaped.gurobienv,"BarHomogeneous",-1)
                lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
                setsolver(lshaped.masterModel,lshaped.masterSolver)
            else
                if lshaped.status == :Infeasible
                    println("Master is infeasible, aborting procedure.")
                else
                    println("Master could not be solved, aborting procedure")
                end
                println("======================")
                return
            end
        end
        lshaped.masterModel.colVal = getsolution(lshaped.internal)[1:lshaped.masterModel.numCols]
        # Update master solution
        updateSolution!(lshaped)
        removeInactive!(lshaped)
        if length(lshaped.violating) <= lshaped.nscenarios
            queueViolated!(lshaped)
        end

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            updateStructuredModel!(lshaped)
            println("Optimal!")
            println("======================")
            break
        end
    end
end
