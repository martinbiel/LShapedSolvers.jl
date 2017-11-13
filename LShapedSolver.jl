mutable struct LShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    # Master
    masterSolver::AbstractLQSolver
    x::AbstractVector
    obj::Real
    obj_hist::AbstractVector

    # Subproblems
    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Cuts
    θs::AbstractVector
    cuts::Vector{AbstractHyperplane}
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer

    # Params
    status::Symbol
    τ::Float64

    function LShapedSolver(m::JuMPModel,x₀::AbstractVector)
        lshaped = new(m)

        if length(x₀) != m.numCols
            throw(ArgumentError(string("Incorrect length of starting guess, has ",length(x₀)," should be ",m.numCols)))
        end

        lshaped.x = x₀
        init(lshaped)

        return lshaped
    end
end
LShapedSolver(m::JuMPModel) = LShapedSolver(m,rand(m.numCols))

function (lshaped::LShapedSolver)()
    println("Starting L-Shaped procedure")
    println("======================")

    println("Main loop")
    println("======================")

    while true
        # Resolve all subproblems at the current optimal solution
        resolveSubproblems!(lshaped)
        updateObjectiveValue!(lshaped)
        push!(lshaped.obj_hist,lshaped.obj)

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            updateStructuredModel!(lshaped)
            println("Optimal!")
            println("Objective value: ", sum(lshaped.subObjectives))
            println("======================")
            break
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
        # Update master solution
        updateSolution!(lshaped)
    end
end

mutable struct AsynchronousLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    # Master
    masterSolver::AbstractLQSolver
    x::AbstractVector
    obj::Real
    obj_hist::AbstractVector

    # Subproblems
    nscenarios::Integer
    subObjectives::AbstractVector

    # Workers
    subworkers::Vector{RemoteChannel}
    masterColumns::Vector{RemoteChannel}
    cutQueue::RemoteChannel
    finished::RemoteChannel

    # Cuts
    θs::AbstractVector
    cuts::Vector{AbstractHyperplane}
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer

    # Params
    status::Symbol
    τ::Float64

    function AsynchronousLShapedSolver(m::JuMPModel,x₀::AbstractVector)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return LShapedSolver(m,x₀)
        end
        lshaped = new(m)

        if length(x₀) != m.numCols
            throw(ArgumentError(string("Incorrect length of starting guess, has ",length(x₀)," should be ",m.numCols)))
        end

        lshaped.x = x₀
        init(lshaped)

        return lshaped
    end
end
AsynchronousLShapedSolver(m::JuMPModel) = AsynchronousLShapedSolver(m,rand(m.numCols))

@traitimpl IsParallel{AsynchronousLShapedSolver}

function Base.show(io::IO, lshaped::AsynchronousLShapedSolver)
    print(io,"AsynchronousLShapedSolver")
end

function (lshaped::AsynchronousLShapedSolver)()
    println("Starting asynchronous L-Shaped procedure\n")
    println("======================")

    finished_workers = Vector{Future}(nworkers())
    println("Start workers")
    for w in workers()
        println("Send work to ",w)
        finished_workers[w-1] = @spawnat w work_on_subproblems(lshaped.subworkers[w-1],lshaped.cutQueue,lshaped.masterColumns[w-1])
        println("Now ",w, " is working")
    end
    println("Main loop")
    println("======================")
    while true
        wait(lshaped.cutQueue)
        println("Cuts are ready")
        while isready(lshaped.cutQueue)
            # Add new cuts from subworkers
            cut = take!(lshaped.cutQueue)
            if !proper(cut)
                println("Subproblem ",cut.id," is unbounded, aborting procedure.")
                println("======================")
                return
            end
            addCut!(lshaped,cut)
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
        # Update master solution
        updateSolution!(lshaped)
        updateObjectiveValue!(lshaped)
        push!(lshaped.obj_hist,lshaped.obj)
        for rx in lshaped.masterColumns
            put!(rx,lshaped.x)
        end

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            map(rx->put!(rx,[]),lshaped.masterColumns)
            map(wait,finished_workers)
            println("Optimal!")
            println("Objective value: ", sum(lshaped.subObjectives))
            println("======================")
            break
        end
    end
end
