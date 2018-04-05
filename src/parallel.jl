# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} = begin
    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,!IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        # Prepare the subproblems
        m = lshaped.structuredmodel
        for i = 1:lshaped.nscenarios
            y₀ = convert(A,rand(subproblem(m,i).numCols))
            push!(lshaped.subproblems,SubProblem(subproblem(m,i),
                                                 parentmodel(scenarioproblems(m)),
                                                 i,
                                                 probability(m,i),
                                                 copy(lshaped.x),
                                                 y₀,
                                                 subsolver))
        end
        lshaped
    end

    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        @unpack κ = lshaped.parameters
        # Partitioning
        (jobsize,extra) = divrem(lshaped.nscenarios,nworkers())
        # One extra to guarantee coverage
        if extra > 0
            jobsize += 1
        end
        # Load initial decision
        put!(lshaped.decisions,1,lshaped.x)
        # Create subproblems on worker processes
        m = lshaped.structuredmodel
        start = 1
        stop = jobsize
        finished_workers = Vector{Future}(nworkers())
        for w in workers()
            lshaped.work[w-1] = RemoteChannel(() -> Channel{Int}(round(Int,10/κ)), w)
            put!(lshaped.work[w-1],1)
            lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            finished_workers[w-1] = load_worker!(scenarioproblems(m),w,lshaped.subworkers[w-1],lshaped.x,start,stop,subsolver)
            if start > lshaped.nscenarios
                continue
            end
            start += jobsize
            stop += jobsize
            stop = min(stop,lshaped.nscenarios)
        end
        map(wait,finished_workers)
        lshaped
    end
end

@define_traitfn IsParallel calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector) = begin
    function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector,NullTrait)
        return lshaped.c⋅x + sum([subproblem.π*subproblem(x) for subproblem in lshaped.subproblems])
    end

    function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector,IsParallel)
        return lshaped.c⋅x + sum(fetch.([@spawnat w+1 calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
    end
end

# Parallel routines #
# ======================================================================== #
mutable struct DecisionChannel{A <: AbstractArray} <: AbstractChannel
    decisions::Dict{Int,A}
    cond_take::Condition
    DecisionChannel(decisions::Dict{Int,A}) where A <: AbstractArray = new{A}(decisions, Condition())
end

function put!(channel::DecisionChannel, t, x)
    channel.decisions[t] = x
    notify(channel.cond_take)
    return channel
end

function take!(channel::DecisionChannel, t)
    x = fetch(channel,t)
    delete!(channel.decisions, t)
    return x
end

isready(channel::DecisionChannel) = length(channel.decisions) > 1
isready(channel::DecisionChannel, t) = haskey(channel.decisions,t)

function fetch(channel::DecisionChannel, t)
    wait(channel,t)
    return channel.decisions[t]
end

function wait(channel::DecisionChannel, t)
    while !isready(channel, t)
        wait(channel.cond_take)
    end
end

SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
ScenarioProblems{D,SD,S} = RemoteChannel{Channel{StochasticPrograms.ScenarioProblems{D,SD,S}}}
Work = RemoteChannel{Channel{Int}}
Decisions{A} = RemoteChannel{DecisionChannel{A}}
QCut{T} = Tuple{Int,T,SparseHyperPlane{T}}
CutQueue{T} = RemoteChannel{Channel{QCut{T}}}

function load_worker!(sp::StochasticPrograms.ScenarioProblems,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      subsolver::AbstractMathProgSolver)
    problems = [sp.problems[i] for i = start:stop]
    πs = [probability(sp.scenariodata[i]) for i = start:stop]
    return remotecall(init_subworker!,
                      w,
                      worker,
                      sp.parent,
                      problems,
                      πs,
                      x,
                      subsolver,
                      collect(start:stop))
end

function load_worker!(sp::StochasticPrograms.DScenarioProblems,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      subsolver::AbstractMathProgSolver)
    return remotecall(init_subworker!,
                      w,
                      worker,
                      sp[w-1],
                      x,
                      subsolver,
                      collect(start:stop))
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         parent::JuMP.Model,
                         submodels::Vector{JuMP.Model},
                         πs::A,
                         x::A,
                         subsolver::AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems = Vector{SubProblem{T,A,S}}(length(ids))
    for (i,id) = enumerate(ids)
        y₀ = convert(A,rand(submodels[i].numCols))
        subproblems[i] = SubProblem(submodels[i],parent,id,πs[i],x,y₀,subsolver)
    end
    put!(subworker,subproblems)
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         scenarioproblems::ScenarioProblems,
                         x::A,
                         subsolver::AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems = Vector{SubProblem{T,A,S}}(length(ids))
    for (i,id) = enumerate(ids)
        y₀ = convert(A,rand(sp.problems[i].numCols))
        subproblems[i] = SubProblem(sp.problems[i],sp.parent,id,probability(sp.scenariodata[i]),x,y₀,subsolver)
    end
    put!(subworker,subproblems)
end

function work_on_subproblems!(subworker::SubWorker{T,A,S},
                              work::Work,
                              cuts::CutQueue{T},
                              decisions::Decisions{A}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    while true
        t::Int = take!(work)
        if t == -1
            # Worker finished
            return
        end
        x::A = fetch(decisions,t)
        update_subproblems!(subproblems,x)
        for subproblem in subproblems
            cut = subproblem()
            Q::T = cut(x)
            try
                put!(cuts,(t,Q,cut))
            catch err
                if err isa InvalidStateException
                    # Master closed the cut channel. Worker finished
                    return
                end
            end
        end
    end
end

function fill_subproblems!(subworker::SubWorker{T,A,S},
                           scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    for (i,submodel) in enumerate(sp.problems)
        snrows, sncols = length(submodel.linconstr), submodel.numCols
        subproblem = subproblems[i]
        submodel.colVal = copy(subproblem.y)
        submodel.redCosts = getreducedcosts(subproblem.solver.lqmodel)
        submodel.linconstrDuals = getconstrduals(subproblem.solver.lqmodel)
        submodel.objVal = getobjval(subproblem.solver)
    end
end

function calculate_subobjective(subworker::SubWorker{T,A,S},
                                x::A) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*subproblem(x) for subproblem in subproblems])
    else
        return zero(T)
    end
end
