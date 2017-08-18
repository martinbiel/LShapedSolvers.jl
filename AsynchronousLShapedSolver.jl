mutable struct AsynchronousLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterSolver::AbstractMathProgSolver
    gurobienv::Gurobi.Env

    readyworkers::RemoteChannel
    subworkers::Vector{RemoteChannel}

    # Regularizer
    a::Vector{Float64}
    Qa::Float64

    # Cuts
    θs
    ready
    cuts::RemoteChannel
    numOptimalityCuts::Integer
    numFeasibilityCuts::Integer

    status::Symbol
    τ::Float64

    function AsynchronousLShapedSolver(m::JuMPModel,a::Vector{Float64})
        lshaped = new(m)

        if length(a) != m.numCols
            throw(ArgumentError(string("Incorrect length of regularizer, has ",length(a)," should be ",m.numCols)))
        end
        lshaped.a = a

        init(lshaped)

        return lshaped
    end
end

@traitimpl IsParallel{AsynchronousLShapedSolver}
@traitimpl IsRegularized{AsynchronousLShapedSolver}

AsynchronousLShapedSolver(m::JuMPModel,a::AbstractVector) = AsynchronousLShapedSolver(m,convert(Vector{Float64},a))

function nscenarios(lshaped::AsynchronousLShapedSolver)
    return sum([length(fetch(subworker)) for subworker in lshaped.subworkers])
end

function init_subworker(subworker::RemoteChannel,parent::JuMPModel,submodels::Vector{JuMPModel},nsubproblems::Integer,πs::AbstractVector)
    subproblems = Vector{SubProblem}(nsubproblems)
    for i = 1:nsubproblems
        subproblems[i] = SubProblem(submodels[i],parent,i,πs[i])
    end
    put!(subworker,subproblems)
end

function work_on_subproblems(subworker::RemoteChannel,cuts::RemoteChannel,readyworkers::RemoteChannel,x::AbstractVector,w::Integer)
    subproblems = fetch(subworker)
    updateSubProblems!(subproblems,x)
    for subprob in subproblems
        println("Solving subproblem: ",subprob.id)
        put!(cuts,subprob())
    end
    put!(readyworkers,w)
end

function Base.show(io::IO, lshaped::AsynchronousLShapedSolver)
    print(io,"AsynchronousLShapedSolver")
end

@traitfn function init{LS <: AbstractLShapedSolver; IsParallel{LS}}(lshaped::LS)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)

    # Master problem (On Master process)
    lshaped.masterModel = extractMaster!(m)
    prepareMaster!(lshaped,n)

    # Workers
    lshaped.readyworkers = RemoteChannel(() -> Channel{Int}(nworkers()),1)
    lshaped.subworkers = Vector{RemoteChannel}(nworkers())
    (jobLength,extra) = divrem(n,nworkers())
    # One extra to guarantee coverage
    if extra > 0
        jobLength += 1
    end

    # Create subproblems on worker processes
    start = 1
    stop = jobLength
    @sync for w in workers()
        lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem}}(1), w)
        submodels = [getchildren(m)[i] for i = start:stop]
        πs = [getprobability(lshaped.structuredModel)[i] for i = start:stop]
        @spawnat w init_subworker(lshaped.subworkers[w-1],m,submodels,stop-start+1,πs)
        start += jobLength
        stop += jobLength
        if stop > n
            stop = n
        end
        put!(lshaped.readyworkers,w)
    end

    lshaped.cuts = RemoteChannel(() -> Channel{AbstractCut}(2*n))
    lshaped.numOptimalityCuts = 0
    lshaped.numFeasibilityCuts = 0

    lshaped.τ = 1e-6
end

function (lshaped::AsynchronousLShapedSolver)()
    println("Starting paralell L-Shaped procedure\n")
    println("======================")
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.status = solve(lshaped.masterModel)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateMasterSolution!(lshaped)

    updatedMaster = true
    addedCut = false

    println("Main loop")
    println("======================")

    while true
        if updatedMaster
            @sync while isready(lshaped.readyworkers)
                # Update outdated workers with the latest master column
                w = take!(lshaped.readyworkers)
                @spawnat w work_on_subproblems(lshaped.subworkers[w-1],lshaped.cuts,lshaped.readyworkers,lshaped.structuredModel.colVal,w)
            end
        end

        if isready(lshaped.cuts)
            while isready(lshaped.cuts)
                # Add new cuts from subworkers
                cut = take!(lshaped.cuts)
                if !proper(cut)
                    println("Subproblem ",subprob.id," is unbounded, aborting procedure.")
                    println("======================")
                    return
                end
                addedCut |= addCut!(lshaped,cut)
                @show addedCut
            end
            # Resolve master problem
            obj = getobjectivevalue(lshaped.structuredModel)
            obj *= lshaped.structuredModel.objSense == :Min ? 1 : -1
            if !addedCut || (obj - lshaped.Qa) <= lshaped.τ
                lshaped.a = lshaped.structuredModel.colVal
                lshaped.Qa = obj
            end

            updateObjective!(lshaped)

            # Resolve master
            println("Solving master problem")
            lshaped.status = solve(lshaped.masterModel)
            if lshaped.status != :Optimal
                setparam!(lshaped.gurobienv,"Presolve",2)
                setparam!(lshaped.gurobienv,"BarHomogeneous",1)
                lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
                setsolver(lshaped.masterModel,lshaped.masterSolver)
                lshaped.status = solve(lshaped.masterModel)
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
            # Update master solution
            updateMasterSolution!(lshaped)
            updatedMaster = true

            if checkOptimality(lshaped)
                # Optimal
                lshaped.status = :Optimal
                println("Optimal!")
                println("======================")
                break
            end

            addedCut = false
        else
            updatedMaster = false
        end
    end
end
