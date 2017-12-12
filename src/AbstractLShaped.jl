abstract type AbstractLShapedSolver{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} end

nscenarios(lshaped::AbstractLShapedSolver) = lshaped.nscenarios

function Base.show(io::IO, lshaped::AbstractLShapedSolver)
    print(io,"LShapedSolver")
end

function Base.show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end

# Initialization #
# ======================================================================== #
function init!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Prepare the master optimization problem
    prepare_master!(lshaped)
    # Finish initialization based on solver traits
    init_solver!(lshaped)
    init_subproblems!(lshaped,subsolver)
end

# ======================================================================== #

# Functions #
# ======================================================================== #
function update_solution!(lshaped::AbstractLShapedSolver)
    ncols = lshaped.structuredmodel.numCols
    x = getsolution(lshaped.mastersolver)
    lshaped.x[1:ncols] = x[1:ncols]
    lshaped.θs[:] = x[end-lshaped.nscenarios+1:end]
    nothing
end

function update_structuredmodel!(lshaped::AbstractLShapedSolver)
    lshaped.structuredmodel.colVal = copy(lshaped.x)
    lshaped.structuredmodel.objVal = lshaped.c⋅lshaped.x + sum(lshaped.subobjectives)

    for i in 1:lshaped.nscenarios
        m = getchildren(lshaped.structuredmodel)[i]
        m.colVal = copy(getsolution(lshaped.subproblems[i].solver))
        m.objVal = getobjval(lshaped.subproblems[i].solver)
    end
    nothing
end

function calculate_estimate(lshaped::AbstractLShapedSolver)
    return lshaped.c⋅lshaped.x + sum(lshaped.θs)
end

function calculate_objective_value(lshaped::AbstractLShapedSolver)
    return lshaped.c⋅lshaped.x + sum(lshaped.subobjectives)
end

function get_solution(lshaped::AbstractLShapedSolver)
    return lshaped.x
end

function get_objective_value(lshaped::AbstractLShapedSolver)
    if !isempty(lshaped.Q_history)
        return lshaped.Q_history[end]
    else
        return calculate_objective_value(lshaped)
    end
end

function prepare_master!(lshaped::AbstractLShapedSolver)
    # θs
    for i = 1:lshaped.nscenarios
        addvar!(lshaped.mastersolver.lqmodel,-Inf,Inf,1.0)
    end
end

function resolve_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S}) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Update subproblems
    update_subproblems!(lshaped.subproblems,lshaped.x)

    # Solve sub problems
    for subproblem ∈ lshaped.subproblems
        println("Solving subproblem: ",subproblem.id)
        cut::SparseHyperPlane{T} = subproblem()
        if !bounded(cut)
            println("Subproblem ",subproblem.id," is unbounded, aborting procedure.")
            println("======================")
            return
        end
        addcut!(lshaped,cut)
    end
end

# Cut functions #
# ======================================================================== #
active(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = active(hyperplane,lshaped.x,lshaped.τ)
active(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = optimal(cut,lshaped.x,lshaped.θs[cut.id],lshaped.τ)
satisfied(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = satisfied(hyperplane,lshaped.x,lshaped.τ)
satisfied(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = satisfied(cut,lshaped.x,lshaped.θs[cut.id],lshaped.τ)
violated(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = !satisfied(lshaped,hyperplane)
gap(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = gap(hyperplane,lshaped.x)
gap(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = gap(cut,lshaped.x,lshaped.θs[cut.id])

function addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut},x::AbstractVector)
    Q = cut(x)
    θ = lshaped.θs[cut.id]
    τ = lshaped.τ

    lshaped.subobjectives[cut.id] = Q

    println("θ",cut.id,": ", θ)
    println("Q",cut.id,": ", Q)

    if θ > -Inf && abs(θ-Q) <= τ*(1+abs(Q))
        # Optimal with respect to this subproblem
        println("Optimal with respect to subproblem ", cut.id)
        return false
    end

    println("Added Optimality Cut")
    if hastrait(lshaped,UsesLocalization)
        push!(lshaped.committee,cut)
    end
    addconstr!(lshaped.mastersolver.lqmodel,lowlevel(cut)...)
    push!(lshaped.cuts,cut)
    return true
end
addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = addcut!(lshaped,cut,lshaped.x)

function addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{FeasibilityCut})
    D = cut.δQ
    d = cut.q

    # Scale to avoid numerical issues
    scaling = abs(d)
    if scaling == 0
        scaling = maximum(D)
    end

    D = D/scaling

    println("Added Feasibility Cut")
    if hastrait(lshaped,UsesLocalization)
        push!(lshaped.committee,cut)
    end
    addconstr!(lshaped.mastersolver.lqmodel,lowlevel(cut)...)
    push!(lshaped.cuts,cut)
    return true
end
# ======================================================================== #

# Parallel routines #
# ======================================================================== #
SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
MasterColumn{A} = RemoteChannel{Channel{A}}
CutQueue{T} = RemoteChannel{Channel{SparseHyperPlane{T}}}

function init_subworker!(subworker::SubWorker{T,A,S},
                         parent::JuMPModel,
                         submodels::Vector{JuMPModel},
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

function work_on_subproblems!(subworker::SubWorker{T,A,S},
                              cuts::CutQueue{T},
                              rx::MasterColumn{A}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    while true
        wait(rx)
        x::A = take!(rx)
        if isempty(x)
            println("Worker finished")
            return
        end
        update_subproblems!(subproblems,x)
        for subproblem in subproblems
            println("Solving subproblem: ",subproblem.id)
            put!(cuts,subproblem())
            println("Subproblem: ",subproblem.id," solved")
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

# ======================================================================== #
# TRAITS #
# ======================================================================== #
# UsesLocalization: Algorithm uses some localization method
@define_trait UsesLocalization = begin
    IsRegularized # Algorithm uses the regularized decomposition method of Ruszczyński
    HasTrustRegion # Algorithm uses the trust-region method of Linderoth/Wright
end

@define_traitfn UsesLocalization function init_solver!(lshaped::AbstractLShapedSolver{T,A,M,S}) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    nothing
end

@define_traitfn UsesLocalization function check_optimality(lshaped::AbstractLShapedSolver)
    Q = sum(lshaped.subobjectives)
    θ = sum(lshaped.θs)
    return θ > -Inf && abs(θ-Q) <= lshaped.τ*(1+abs(θ))
end function check_optimality(lshaped::AbstractLShapedSolver,UsesLocalization)
    θ = calculate_estimate(lshaped)
    Q = lshaped.solverdata.Q̃
    if θ > -Inf && Q < Inf && abs(θ - lshaped.solverdata.Q̃) <= lshaped.τ*(1+abs(lshaped.solverdata.Q̃))
        return true
    else
        return false
    end
end

@define_traitfn UsesLocalization take_step!(lshaped::AbstractLShapedSolver)

@define_traitfn UsesLocalization remove_inactive!(lshaped::AbstractLShapedSolver) function remove_inactive!(lshaped::AbstractLShapedSolver,UsesLocalization)
    inactive = find(c->!active(lshaped,c),lshaped.committee)
    diff = length(lshaped.committee) - length(lshaped.structuredmodel.linconstr) - lshaped.nscenarios
    if isempty(inactive) || diff <= 0
        return false
    end
    if diff <= length(inactive)
        inactive = inactive[1:diff]
    end
    append!(lshaped.inactive,lshaped.committee[inactive])
    deleteat!(lshaped.committee,inactive)
    delconstrs!(lshaped.mastersolver.lqmodel,inactive)
    return true
end

@define_traitfn UsesLocalization queueViolated!(lshaped::AbstractLShapedSolver) function queueViolated!(lshaped::AbstractLShapedSolver,UsesLocalization)
    violating = find(c->violated(lshaped,c),lshaped.inactive)
    if isempty(violating)
        return false
    end
    gaps = map(c->gap(lshaped,c),lshaped.inactive[violating])
    for (c,g) in zip(lshaped.inactive[violating],gaps)
        enqueue!(lshaped.violating,c,g)
    end
    deleteat!(lshaped.inactive,violating)
    return true
end

# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Prepare the subproblems
    m = lshaped.structuredmodel
    π = getprobability(m)
    for i = 1:lshaped.nscenarios
        y₀ = convert(A,rand(getchildren(m)[i].numCols))
        push!(lshaped.subproblems,SubProblem(getchildren(m)[i],m,i,π[i],copy(lshaped.x),y₀,subsolver))
    end
    lshaped
end

@implement_traitfn IsParallel function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Partitioning
    (jobLength,extra) = divrem(lshaped.nscenarios,nworkers())
    # One extra to guarantee coverage
    if extra > 0
        jobLength += 1
    end
    # Create subproblems on worker processes
    m = lshaped.structuredmodel
    start = 1
    stop = jobLength
    @sync for w in workers()
        lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
        lshaped.mastercolumns[w-1] = RemoteChannel(() -> Channel{A}(10), w)
        put!(lshaped.mastercolumns[w-1],lshaped.x)
        submodels = [getchildren(m)[i] for i = start:stop]
        πs = [getprobability(m)[i] for i = start:stop]
        @spawnat w init_subworker!(lshaped.subworkers[w-1],
                                   m,
                                   submodels,
                                   πs,
                                   lshaped.x,
                                   subsolver,
                                   collect(start:stop))
        if start > lshaped.nscenarios
            continue
        end
        start += jobLength
        stop += jobLength
        stop = min(stop,lshaped.nscenarios)
    end
    lshaped
end

@define_traitfn IsParallel function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector)
    return lshaped.c⋅x + sum([subproblem.π*subproblem(x) for subproblem in lshaped.subproblems])
end

@implement_traitfn IsParallel function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector)
    c = lshaped.structuredmodel.obj.aff.coeffs
    objidx = [v.col for v in lshaped.structuredmodel.obj.aff.vars]
    return c⋅x[objidx] + sum(fetch.([@spawnat w calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
end

# ------------------------------------------------------------------------ #

# ======================================================================== #
