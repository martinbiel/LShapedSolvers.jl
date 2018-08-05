# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} = begin
    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,!IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        # Prepare the subproblems
        m = lshaped.structuredmodel
        load_subproblems!(lshaped,scenarioproblems(m),subsolver)
        append!(lshaped.subobjectives,zeros(nbundles(lshaped)))
        return lshaped
    end

    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        @unpack κ = lshaped.parameters
        # Partitioning
        (jobsize,extra) = divrem(nscenarios(lshaped),nworkers())
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
            lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            active_workers[w-1] = load_worker!(scenarioproblems(m),w,lshaped.subworkers[w-1],lshaped.x,start,stop,subsolver)
            if start > lshaped.nscenarios
                continue
            end
            start += jobsize
            stop += jobsize
            stop = min(stop,lshaped.nscenarios)
        end
        for w in workers()
            lshaped.work[w-1] = RemoteChannel(() -> Channel{Int}(round(Int,10/κ)), w)
            put!(lshaped.work[w-1],1)
        end
        # Prepare memory
        push!(lshaped.subobjectives,zeros(nbundles(lshaped)))
        push!(lshaped.finished,0)
        log_val = lshaped.parameters.log
        lshaped.parameters.log = false
        log!(lshaped)
        lshaped.parameters.log = log_val
        # Ensure initialization is finished
        map(wait,active_workers)
        return lshaped
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

@define_traitfn IsParallel nbundles(lshaped::AbstractLShapedSolver) = begin
    function nbundles(lshaped::AbstractLShapedSolver,!IsParallel)
        (n,extra) = divrem(lshaped.nscenarios,lshaped.parameters.bundle)
        if extra > 0
            n += 1
        end
        return n
    end

    function nbundles(lshaped::AbstractLShapedSolver,IsParallel)
        (jobsize,extra) = divrem(lshaped.nscenarios,nworkers())
        if extra > 0
            jobsize += 1
        end
        (n,extra) = divrem(jobsize,lshaped.parameters.bundle)
        if extra > 0
            n += 1
        end
        remainder = lshaped.nscenarios-(nworkers()-1)*jobsize
        (bundlerem,extra) = divrem(remainder,lshaped.parameters.bundle)
        if extra > 0
            bundlerem += 1
        end
        return n*(nworkers()-1)+bundlerem
    end
end

@define_traitfn IsParallel init_workers!(lshaped::AbstractLShapedSolver) = begin
    function init_workers!(lshaped::AbstractLShapedSolver,IsParallel)
        bundleindex = 1
        (jobsize,extra) = divrem(lshaped.nscenarios,nworkers())
        if extra > 0
            jobsize += 1
        end
        (n,extra) = divrem(jobsize,lshaped.parameters.bundle)
        if extra > 0
            n += 1
        end
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            if lshaped.parameters.bundle == 1
                active_workers[w-1] = remotecall(work_on_subproblems!,
                                                 w,
                                                 lshaped.subworkers[w-1],
                                                 lshaped.work[w-1],
                                                 lshaped.cutqueue,
                                                 lshaped.decisions)
            else
                active_workers[w-1] = remotecall(work_on_subproblems!,
                                                 w,
                                                 lshaped.subworkers[w-1],
                                                 lshaped.work[w-1],
                                                 lshaped.cutqueue,
                                                 lshaped.decisions,
                                                 lshaped.parameters.bundle,
                                                 bundleindex)
                bundleindex += n
            end
        end
        return active_workers
    end
end

@define_traitfn IsParallel close_workers!(lshaped::AbstractLShapedSolver,workers::Vector{Future}) = begin
    function close_workers!(lshaped::AbstractLShapedSolver,workers::Vector{Future},IsParallel)
        @async begin
            map((w)->close(w),lshaped.work)
            map(wait,workers)
        end
    end
end

@define_traitfn IsParallel calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector) = begin
    function calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector,!IsParallel)
        return lshaped.c⋅x + sum([subproblem.π*subproblem(x) for subproblem in lshaped.subproblems])
    end

    function calculate_objective_value(lshaped::AbstractLShapedSolver,x::AbstractVector,IsParallel)
        return lshaped.c⋅x + sum(fetch.([@spawnat w+1 calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
    end
end

@define_traitfn IsParallel fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems) = begin
    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,!IsParallel)
        for (i,submodel) in enumerate(scenarioproblems.problems)
            fill_submodel!(submodel,lshaped.subproblems[i])
        end
    end

    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,IsParallel)
        j = 0
        for w in workers()
            n = remotecall_fetch((sw)->length(fetch(sw)),w,lshaped.subworkers[w-1])
            for i = 1:n
                fill_submodel!(scenarioproblems.problems[i+j],remotecall_fetch((sw,i,x)->begin
                                                                               sp = fetch(sw)[i]
                                                                               sp(x)
                                                                               get_solution(sp)
                                                                               end,
                                                                               w,
                                                                               lshaped.subworkers[w-1],
                                                                               i,
                                                                               lshaped.x)...)
            end
            j += n
        end
    end
end

@define_traitfn IsParallel fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems) = begin
    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,!IsParallel)
        active_workers = Vector{Future}(nworkers())
        j = 1
        for w in workers()
            n = remotecall_fetch((sp)->length(fetch(sp).problems),w,scenarioproblems[w-1])
            for i in 1:n
                active_workers[j] = remotecall((sp,i,x,μ,λ,C) -> fill_submodel!(fetch(sp).problems[i],x,μ,λ,C),
                                               w,
                                               scenarioproblems[w-1],
                                               i,
                                               get_solution(lshaped.subproblems[j])...)
                j += 1
            end
        end
        @async map(wait,active_workers)
    end

    function fill_submodels!(lshaped::AbstractLShapedSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,IsParallel)
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(fill_submodels!,
                                             w,
                                             lshaped.subworkers[w-1],
                                             lshaped.x,
                                             scenarioproblems[w-1])
        end
        @async map(wait,active_workers)
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
    channel.decisions[t] = copy(x)
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
    if isempty(subproblems)
       # Workers has nothing do to, return.
       return
    end
    while true
        t::Int = try
            wait(work)
            take!(work)
        catch err
            if err isa InvalidStateException
                # Master closed the work channel. Worker finished
                return
            end
        end
        if t == -1
            # Worker finished
            return
        end
        x::A = fetch(decisions,t)
        for subproblem in subproblems
            @schedule begin
                update_subproblem!(subproblem,x)
                cut = subproblem()
                Q::T = cut(x)
                put!(cuts,(t,Q,cut))
            end
        end
    end
end

function work_on_subproblems!(subworker::SubWorker{T,A,S},
                              work::Work,
                              cuts::CutQueue{T},
                              decisions::Decisions{A},
                              bundle::Int,
                              bundleindex::Int) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    while true
        t::Int = try
            wait(work)
            take!(work)
        catch err
            if err isa InvalidStateException
                # Master closed the work channel. Worker finished
                return
            end
        end
        if t == -1
            # Worker finished
            return
        end
        if isempty(subproblems)
            # Workers has nothing do to
            continue
        end
        x::A = fetch(decisions,t)
        (njobs,extra) = divrem(length(subproblems),bundle)
        if extra > 0
            njobs += 1
        end
        for i = 1:njobs
            @schedule begin
                bundled_cuts = Vector{SparseHyperPlane{T}}()
                Qagg = zero(T)
                for subproblem in subproblems[(i-1)*bundle+1:min(i*bundle,length(subproblems))]
                    update_subproblem!(subproblem,x)
                    cut = subproblem()
                    Qagg += cut(x)
                    push!(bundled_cuts,cut)
                end
                put!(cuts,(t,Qagg,aggregate!(bundled_cuts,bundleindex+i-1)))
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
                         x::A,
                         scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    for (i,submodel) in enumerate(sp.problems)
        subproblems[i](x)
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
            map(w->put!(w,-1),lshaped.work)
            warn("Subproblem ",cut.id," is unbounded, aborting procedure.")
            return :Unbounded
        end
        addcut!(lshaped,cut,Q)
        lshaped.subobjectives[t][cut.id] = Q
        lshaped.finished[t] += 1
        if lshaped.finished[t] == nbundles(lshaped)
            lshaped.solverdata.timestamp = t
            lshaped.x[:] = fetch(lshaped.decisions,t)
            lshaped.Q_history[t] = current_objective_value(lshaped,lshaped.subobjectives[t])
            lshaped.solverdata.Q = lshaped.Q_history[t]
            lshaped.solverdata.θ = t > 1 ? lshaped.θ_history[t-1] : -1e10
            take_step!(lshaped)
            lshaped.solverdata.θ = lshaped.θ_history[t]
            # Check if optimal
            if check_optimality(lshaped)
                # Optimal, tell workers to stop
                map(w->put!(w,t),lshaped.work)
                map(w->put!(w,-1),lshaped.work)
                # Final log
                log!(lshaped,lshaped.solverdata.iterations)
                return :Optimal
            end
        end
    end
    # Resolve master
    t = lshaped.solverdata.iterations
    if lshaped.finished[t] >= lshaped.parameters.κ*nbundles(lshaped) && length(lshaped.cuts) >= nbundles(lshaped)
        try
            solve_problem!(lshaped,lshaped.mastersolver)
        catch
            # Master problem could not be solved for some reason.
            @unpack Q,θ = lshaped.solverdata
            gap = abs(θ-Q)/(abs(Q)+1e-10)
            warn("Master problem could not be solved, solver returned status $(status(lshaped.mastersolver)). The following relative tolerance was reached: $(@sprintf("%.1e",gap)). Aborting procedure.")
            map(w->put!(w,-1),lshaped.work)
            return :StoppedPrematurely
        end
        if status(lshaped.mastersolver) == :Infeasible
            warn("Master is infeasible. Aborting procedure.")
            map(w->put!(w,-1),lshaped.work)
            return :Infeasible
        end
        # Update master solution
        update_solution!(lshaped)
        θ = calculate_estimate(lshaped)
        if t > 1 && abs(θ-lshaped.θ_history[t-1]) <= 10*lshaped.parameters.τ*abs(1e-10+θ) && lshaped.finished[t] != nbundles(lshaped)
            # Not enough new information in master. Repeat iterate
            return :Valid
        end
        lshaped.solverdata.θ = θ
        lshaped.θ_history[t] = θ
        # Project (if applicable)
        project!(lshaped)
        # If all work is finished at this timestamp, check optimality
        if lshaped.finished[t] == nbundles(lshaped)
            # Check if optimal
            if check_optimality(lshaped)
                # Optimal, tell workers to stop
                map(w->put!(w,t),lshaped.work)
                map(w->put!(w,-1),lshaped.work)
                # Final log
                log!(lshaped,t)
                return :Optimal
            end
        end
        # Log progress at current timestamp
        log_regularization!(lshaped,t)
        # Send new decision vector to workers
        put!(lshaped.decisions,t+1,lshaped.x)
        for w in lshaped.work
            put!(w,t+1)
        end
        # Prepare memory for next iteration
        push!(lshaped.subobjectives,zeros(nbundles(lshaped)))
        push!(lshaped.finished,0)
        # Log progress
        log!(lshaped)
        lshaped.θ_history[t+1] = -Inf
    end
    # Just return a valid status for this iteration
    return :Valid
end
