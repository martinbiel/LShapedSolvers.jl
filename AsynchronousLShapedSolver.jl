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
    readyworkers::RemoteChannel
    subworkers::Vector{RemoteChannel}
    cutQueue::RemoteChannel
    mastervector::RemoteChannel

    # Cuts
    θs::AbstractVector
    cuts::Vector{AbstractHyperplane}
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer

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

function init_subworker(subworker::RemoteChannel,parent::JuMPModel,submodels::Vector{JuMPModel},πs::AbstractVector,ids::AbstractVector)
    subproblems = Vector{SubProblem}(length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(submodels[i],parent,id,πs[i])
    end
    put!(subworker,subproblems)
end

function work_on_subproblems(subworker::RemoteChannel,cuts::RemoteChannel,readyworkers::RemoteChannel,rx::RemoteChannel)
    subproblems = fetch(subworker)
    updateSubProblems!(subproblems,fetch(rx))
    for subprob in subproblems
        println("Solving subproblem: ",subprob.id)
        put!(cuts,subprob())
        println("Subproblem: ",subprob.id," solved")
    end
    put!(readyworkers,myid())
end

function calculate_subobjective(subworker::RemoteChannel,x::AbstractVector)
    subproblems = fetch(subworker)
    if length(subproblems) > 0
        return sum([subprob.π*subprob(x) for subprob in subproblems])
    else
        return zero(eltype(x))
    end
end

function Base.show(io::IO, lshaped::AsynchronousLShapedSolver)
    print(io,"AsynchronousLShapedSolver")
end

@traitfn function init{LS <: AbstractLShapedSolver; IsParallel{LS}}(lshaped::LS)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)
    lshaped.nscenarios = n

        # Initialize variables specific to traits
    if istrait(IsRegularized{LS})
        lshaped.Q̃_hist = Float64[]
        lshaped.σ = 1.0
        lshaped.γ = 0.9
        lshaped.Δ̅ = max(1.0,0.2*norm(lshaped.ξ,Inf))
        lshaped.Δ̅_hist = [lshaped.Δ̅]

        lshaped.nExactSteps = 0
        lshaped.nApproximateSteps = 0
        lshaped.nNullSteps = 0

        lshaped.committee = linearconstraints(lshaped.structuredModel)
        lshaped.inactive = Vector{AbstractHyperplane}()
        lshaped.violating = PriorityQueue(Reverse)
    elseif istrait(HasTrustRegion{LS})
        lshaped.Q̃_hist = Float64[]
        lshaped.Δ = max(1.0,0.2*norm(lshaped.ξ,Inf))
        lshaped.Δ_hist = [lshaped.Δ]
        lshaped.Δ̅ = 1000*lshaped.Δ
        lshaped.cΔ = 0
        lshaped.γ = 1e-4

        lshaped.nMajorSteps = 0
        lshaped.nMinorSteps = 0

        lshaped.committee = Vector{AbstractHyperplane}()
        #lshaped.committee = linearconstraints(lshaped.structuredModel)
        lshaped.inactive = Vector{AbstractHyperplane}()
        lshaped.violating = PriorityQueue(Reverse)
    else
        lshaped.obj_hist = Float64[]
    end

    # Master problem (On Master process)
    prepareMaster!(lshaped,n)
    lshaped.θs = fill(-Inf,n)
    lshaped.obj = Inf

    # Workers
    lshaped.readyworkers = RemoteChannel(() -> Channel{Int}(nworkers()),1)
    lshaped.subworkers = Vector{RemoteChannel}(nworkers())
    lshaped.cutQueue = RemoteChannel(() -> Channel{AbstractHyperplane}(2*n))
    lshaped.mastervector = RemoteChannel(() -> Channel{AbstractVector}(1),1)
    put!(lshaped.mastervector,lshaped.x)
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
        @spawnat w init_subworker(lshaped.subworkers[w-1],m,submodels,πs,collect(start:stop))
        if start > n
            continue
        else
            put!(lshaped.readyworkers,w)
        end
        start += jobLength
        stop += jobLength
        stop = min(stop,n)
    end
    lshaped.subObjectives = zeros(n)

    lshaped.cuts = Vector{AbstractHyperplane}()
    lshaped.nOptimalityCuts = 0
    lshaped.nFeasibilityCuts = 0

    lshaped.τ = 1e-6
end

@traitfn function calculateObjective{LS <: AbstractLShapedSolver; IsParallel{LS}}(lshaped::LS,x::AbstractVector)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objidx = [v.col for v in lshaped.structuredModel.obj.aff.vars]

    return c⋅x[objidx] + sum(fetch.([@spawnat w calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
end

function (lshaped::AsynchronousLShapedSolver)()
    println("Starting asynchronous L-Shaped procedure\n")
    println("======================")

    updatedMaster = true
    addedCut = false
    encountered = IntSet()

    println("Main loop")
    println("======================")

    while true
        @sync while isready(lshaped.readyworkers) && updatedMaster
            println("Prepare for work")
            w = take!(lshaped.readyworkers)
            if w in encountered
                println("Already encountered worker ",w)
                put!(lshaped.readyworkers,w)
                break
            end
            # Update outdated workers with the latest master column
            push!(encountered,w)
            println("Send work to ",w)
            @spawnat w work_on_subproblems(lshaped.subworkers[w-1],lshaped.cutQueue,lshaped.readyworkers,lshaped.mastervector)
        end
        println("SUP")
        if isready(lshaped.cutQueue)
            println("Cuts are ready")
            empty!(encountered)
            while isready(lshaped.cutQueue)
                # Add new cuts from subworkers
                cut = take!(lshaped.cutQueue)
                if !proper(cut)
                    println("Subproblem ",cut.id," is unbounded, aborting procedure.")
                    println("======================")
                    return
                end
                addedCut |= addCut!(lshaped,cut)
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
            take!(lshaped.mastervector)
            put!(lshaped.mastervector,lshaped.x)
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
