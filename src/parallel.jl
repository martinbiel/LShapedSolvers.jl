# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} = begin
    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,!IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        # Prepare the subproblems
        m = lshaped.structuredmodel
        load_subproblems!(lshaped,scenarioproblems(m),subsolver)
        return lshaped
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
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            lshaped.work[w-1] = RemoteChannel(() -> Channel{Int}(round(Int,10/κ)), w)
            put!(lshaped.work[w-1],1)
            lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            active_workers[w-1] = load_worker!(scenarioproblems(m),w,lshaped.subworkers[w-1],lshaped.x,start,stop,subsolver)
            if start > lshaped.nscenarios
                continue
            end
            start += jobsize
            stop += jobsize
            stop = min(stop,lshaped.nscenarios)
        end
        map(wait,active_workers)
        lshaped
    end
end

@define_traitfn IsParallel iterate!(lshaped::AbstractLShapedSolver) = begin
    function iterate!(lshaped::AbstractLShapedSolver,!IsParallel)
        iterate_nominal!(lshaped)
    end

    function iterate!(lshaped::AbstractLShapedSolver,IsParallel)
        iterate_parallel!(lshaped)
    end
end

@define_traitfn IsParallel init_workers!(lshaped::AbstractLShapedSolver) = begin
    function init_workers!(lshaped::AbstractLShapedSolver,IsParallel)
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(work_on_subproblems!,
                                             w,
                                             lshaped.subworkers[w-1],
                                             lshaped.work[w-1],
                                             lshaped.cutqueue,
                                             lshaped.decisions)
        end
        return active_workers
    end
end

@define_traitfn IsParallel close_workers!(lshaped::AbstractLShapedSolver,workers::Vector{Future}) = begin
    function close_workers!(lshaped::AbstractLShapedSolver,workers::Vector{Future},IsParallel)
        @async begin
            close(lshaped.cutqueue)
            map(wait,workers)
        end
    end
end

@define_traitfn IsParallel calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector) = begin
    function calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector,NullTrait)
        return lshaped.c⋅x + sum([subproblem.π*subproblem(x) for subproblem in lshaped.subproblems])
    end

    function calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector,IsParallel)
        return lshaped.c⋅x + sum(fetch.([@spawnat w+1 calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
    end
end

@define_traitfn IsParallel fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems) = begin
    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,NullTrait)
        for (i,submodel) in enumerate(scenarioproblems.problems)
            fill_submodel!(submodel,lshaped.subproblems[i])
        end
    end

    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,IsParallel)
        j = 0
        for w = 1:length(lshaped.subworkers)
            n = remotecall_fetch((sw)->length(fetch(sw)),w+1,lshaped.subworkers[w])
            for i = 1:n
                fill_submodel!(scenarioproblems.problems[i+j],remotecall_fetch((sw,i)->get_solution(fetch(sw)[i]),w+1,lshaped.subworkers[w],i)...)
            end
            j += n
        end
    end
end

@define_traitfn IsParallel fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems) = begin
    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,NullTrait)
        active_workers = Vector{Future}(length(scenarioproblems))
        j = 1
        for w = 1:length(scenarioproblems)
            n = remotecall_fetch((sp)->length(fetch(sp).problems),w+1,scenarioproblems[w])
            active_workers[w] = remotecall((subproblems,sp) -> begin
                                               scenarioproblems = fetch(sp)
                                               for (i,submodel) in enumerate(scenarioproblems.problems)
                                                 fill_submodel!(submodel,subproblems[i])
                                               end
                                             end,
                                             w+1,
                                             lshaped.subproblems[j:n],
                                             scenarioproblems[w])
            j += n
        end
    end

    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,IsParallel)
        active_workers = Vector{Future}(length(scenarioproblems))
        for w = 1:length(scenarioproblems)
            active_workers[w] = remotecall(fill_submodels!,
                                             w+1,
                                             lshaped.subworkers[w],
                                             scenarioproblems[w])
        end
        map(wait,active_workers)
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

function load_subproblems!(lshaped::AbstractLShapedSolver{T,A},scenarioproblems::StochasticPrograms.ScenarioProblems,subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector}
    for i = 1:lshaped.nscenarios
        m = subproblem(scenarioproblems,i)
        y₀ = convert(A,rand(m.numCols))
        push!(lshaped.subproblems,SubProblem(m,
                                             parentmodel(scenarioproblems),
                                             i,
                                             probability(scenario(scenarioproblems,i)),
                                             copy(lshaped.x),
                                             y₀,
                                             subsolver))
    end
    return lshaped
end

function load_subproblems!(lshaped::AbstractLShapedSolver{T,A},scenarioproblems::StochasticPrograms.DScenarioProblems,subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector}
    for i = 1:lshaped.nscenarios
        m = subproblem(scenarioproblems,i)
        y₀ = convert(A,rand(m.numCols))
        push!(lshaped.subproblems,SubProblem(m,
                                             i,
                                             probability(scenario(scenarioproblems,i)),
                                             copy(lshaped.x),
                                             y₀,
                                             masterterms(scenarioproblems,i),
                                             subsolver))
    end
    return lshaped
end

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

function calculate_subobjective(subworker::SubWorker{T,A,S},
                                x::A) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*subproblem(x) for subproblem in subproblems])
    else
        return zero(T)
    end
end

function fill_submodels!(subworker::SubWorker{T,A,S},
                         scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    for (i,submodel) in enumerate(sp.problems)
        fill_submodel!(submodel,subproblems[i])
    end
end

function fill_submodel!(submodel::JuMP.Model,x::AbstractVector,μ::AbstractVector,λ::AbstractVector,C::Real)
    submodel.colVal = x
    submodel.redCosts = μ
    submodel.linconstrDuals = λ
    submodel.objVal = C
    submodel.objVal *= submodel.objSense == :Min ? 1 : -1
end

function fill_submodel!(submodel::JuMP.Model,subproblem::SubProblem)
    fill_submodel!(submodel,get_solution(subproblem)...)
end

function iterate_parallel!(lshaped::AbstractLShapedSolver{T,A,M,S}) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    wait(lshaped.cutqueue)
    while isready(lshaped.cutqueue)
        # Add new cuts from subworkers
        t::Int,Q::T,cut::SparseHyperPlane{T} = take!(lshaped.cutqueue)
        if !bounded(cut)
            warn("Subproblem ",cut.id," is unbounded, aborting procedure.")
            return :Unbounded
        end
        addcut!(lshaped,cut,Q)
        lshaped.subobjectives[t][cut.id] = Q
        lshaped.finished[t] += 1
        if lshaped.finished[t] == lshaped.nscenarios
            lshaped.Q_history[t] = current_objective_value(lshaped,lshaped.subobjectives[t])
            if lshaped.Q_history[t] <= lshaped.solverdata.Q
                lshaped.solverdata.Q = lshaped.Q_history[t]
            end
            lshaped.x[:] = fetch(lshaped.decisions,t)
            take_step!(lshaped)
        end
    end
    # Resolve master
    t = lshaped.solverdata.timestamp
    if lshaped.finished[t] >= lshaped.parameters.κ*lshaped.nscenarios && length(lshaped.cuts) >= lshaped.nscenarios
        try
            lshaped.mastersolver(lshaped.mastervector)
        catch
            # Master problem could not be solved for some reason.
            warn("Master problem could not be solved. Returned $(status(lshaped.mastersolver)). Aborting procedure.")
            return :StoppedPrematurely
        end
        if status(lshaped.mastersolver) == :Infeasible
            warn("Master is infeasible, aborting procedure.")
            map(w->put!(w,-1),lshaped.work)
            return :Infeasible
        end
        # Update master solution
        update_solution!(lshaped)
        lshaped.solverdata.θ = calculate_estimate(lshaped)
        # Log progress at current timestamp
        log!(lshaped,t)
        # Check if optimal
        if check_optimality(lshaped)
            # Optimal
            map(w->put!(w,-1),lshaped.work)
            lshaped.solverdata.Q = calculate_objective_value(lshaped,lshaped.x)
            lshaped.Q_history[t] = lshaped.solverdata.Q
            return :Optimal
        end
        # Send new decision vector to workers
        put!(lshaped.decisions,t+1,lshaped.x)
        for w in lshaped.work
            put!(w,t+1)
        end
        # Prepare memory for next timestamp
        lshaped.solverdata.timestamp += 1
        push!(lshaped.subobjectives,zeros(lshaped.nscenarios))
        push!(lshaped.finished,0)
        # Log progress
        log!(lshaped)
    end
    # Just return a valid status for this iteration
    return :Valid
end
