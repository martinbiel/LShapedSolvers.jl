abstract type AbstractLShapedSolver end

nscenarios(lshaped::AbstractLShapedSolver) = lshaped.nscenarios

function Base.show(io::IO, lshaped::AbstractLShapedSolver)
    print(io,"LShapedSolver")
end

function Base.show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end

# Initialization #
# ======================================================================== #
function init(lshaped::AbstractLShapedSolver)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)
    lshaped.nscenarios = n

    # Prepare the master optimization problem
    prepareMaster!(lshaped)
    lshaped.θs = fill(-Inf,lshaped.nscenarios)
    lshaped.obj = Inf

    initSolverData!(lshaped)
    initSolver!(lshaped)

    lshaped.subObjectives = zeros(lshaped.nscenarios)

    lshaped.cuts = Vector{AbstractHyperplane}()
    lshaped.nOptimalityCuts = 0
    lshaped.nFeasibilityCuts = 0

    # Set the tolerance
    lshaped.τ = 1e-6
end

# ======================================================================== #

# Functions #
# ======================================================================== #
function updateSolution!(lshaped::AbstractLShapedSolver)
    lshaped.x[1:lshaped.structuredModel.numCols] = lshaped.masterSolver.x[1:lshaped.structuredModel.numCols]
    lshaped.θs = lshaped.masterSolver.x[end-lshaped.nscenarios+1:end]
end

function updateObjectiveValue!(lshaped::AbstractLShapedSolver)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1

    lshaped.obj = c⋅lshaped.x + sum(lshaped.subObjectives)
end

function updateStructuredModel!(lshaped::AbstractLShapedSolver)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    lshaped.structuredModel.colVal = copy(lshaped.x)
    lshaped.structuredModel.objVal = c⋅lshaped.x + sum(lshaped.subObjectives)
    lshaped.structuredModel.objVal *= lshaped.structuredModel.objSense == :Min ? 1 : -1

    for i in 1:lshaped.nscenarios
        m = getchildren(lshaped.structuredModel)[i]
        m.colVal = copy(lshaped.subProblems[i].solver.x)
        m.objVal = copy(lshaped.subProblems[i].solver.obj)
        m.objVal *= m.objSense == :Min ? 1 : -1
    end
end

function extractMaster!(lshaped::AbstractLShapedSolver,src::JuMPModel)
    @assert haskey(src.ext,:Stochastic) "The provided model is not structured"

    # Minimal copy of master part of structured problem
    master = Model()

    if src.colNames[1] == ""
        for varFamily in src.dictList
            JuMP.fill_var_names(JuMP.REPLMode,src.colNames,varFamily)
        end
    end

    # Objective
    master.obj = copy(src.obj, master)
    master.objSense = src.objSense

    # Constraint
    master.linconstr  = map(c->copy(c, master), src.linconstr)

    # Variables
    master.numCols = src.numCols
    master.colNames = src.colNames[:]
    master.colNamesIJulia = src.colNamesIJulia[:]
    master.colLower = src.colLower[:]
    master.colUpper = src.colUpper[:]
    master.colCat = src.colCat[:]
    master.colVal = src.colVal[:]

    lshaped.masterModel = master
end

function prepareMaster!(lshaped::AbstractLShapedSolver)
    lshaped.masterSolver = LQSolver(lshaped.structuredModel)

    # θs
    for i = 1:lshaped.nscenarios
        addvar!(lshaped.masterSolver.model,-Inf,Inf,1.0)
    end
    append!(lshaped.masterSolver.x,zeros(lshaped.nscenarios))
end

function resolveSubproblems!(lshaped::AbstractLShapedSolver)
    # Update subproblems
    updateSubProblems!(lshaped.subProblems,lshaped.x)

    # Solve sub problems
    for subprob in lshaped.subProblems
        println("Solving subproblem: ",subprob.id)
        cut = subprob()
        if !proper(cut)
            println("Subproblem ",subprob.id," is unbounded, aborting procedure.")
            println("======================")
            return
        end
        addCut!(lshaped,cut)
    end
end
# ======================================================================== #

# Parallel routines #
# ======================================================================== #
function init_subworker(subworker::RemoteChannel,parent::JuMPModel,submodels::Vector{JuMPModel},πs::AbstractVector,ids::AbstractVector)
    subproblems = Vector{SubProblem}(length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(submodels[i],parent,id,πs[i])
    end
    put!(subworker,subproblems)
end

function work_on_subproblems(subworker::RemoteChannel,cuts::RemoteChannel,rx::RemoteChannel)
    subproblems = fetch(subworker)
    while true
        wait(rx)
        x = take!(rx)
        if isempty(x)
            println("Worker finished")
            return
        end
        updateSubProblems!(subproblems,x)
        for subprob in subproblems
            println("Solving subproblem: ",subprob.id)
            put!(cuts,subprob())
            println("Subproblem: ",subprob.id," solved")
        end
    end
end

function calculate_subobjective(subworker::RemoteChannel,x::AbstractVector)
    subproblems = fetch(subworker)
    if length(subproblems) > 0
        return sum([subprob.π*subprob(x) for subprob in subproblems])
    else
        return zero(eltype(x))
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

@define_traitfn UsesLocalization function initSolverData!(lshaped::AbstractLShapedSolver)
    lshaped.obj_hist = Float64[]
end

@define_traitfn UsesLocalization function checkOptimality(lshaped::AbstractLShapedSolver)
    Q = sum(lshaped.subObjectives)
    θ = sum(lshaped.θs)
    return θ > -Inf && abs(θ-Q) <= lshaped.τ*(1+abs(θ))
end function checkOptimality(lshaped::AbstractLShapedSolver,UsesLocalization)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    θ = c⋅lshaped.x + sum(lshaped.θs)

    if abs(θ - lshaped.Q̃) <= lshaped.τ*(1+abs(lshaped.Q̃))
        return true
    else
        return false
    end
end

@define_traitfn UsesLocalization queueViolated!(lshaped::AbstractLShapedSolver) function queueViolated!(lshaped::AbstractLShapedSolver,UsesLocalization)
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

# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel function initSolver!(lshaped::AbstractLShapedSolver)
    # Prepare the subproblems
    m = lshaped.structuredModel
    lshaped.subProblems = Vector{SubProblem}(lshaped.nscenarios)
    π = getprobability(m)
    for i = 1:lshaped.nscenarios
        lshaped.subProblems[i] = SubProblem(getchildren(m)[i],m,i,π[i])
    end
    lshaped
end

@implement_traitfn IsParallel function initSolver!(lshaped::AbstractLShapedSolver)
    # Workers
    lshaped.subworkers = Vector{RemoteChannel}(nworkers())
    lshaped.cutQueue = RemoteChannel(() -> Channel{AbstractHyperplane}(4*nworkers()*lshaped.nscenarios))
    lshaped.masterColumns = Vector{RemoteChannel}(nworkers())
    (jobLength,extra) = divrem(lshaped.nscenarios,nworkers())
    # One extra to guarantee coverage
    if extra > 0
        jobLength += 1
    end

    # Create subproblems on worker processes
    start = 1
    stop = jobLength
    @sync for w in workers()
        lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem}}(1), w)
        lshaped.masterColumns[w-1] = RemoteChannel(() -> Channel{AbstractVector}(5), w)
        put!(lshaped.masterColumns[w-1],lshaped.x)
        submodels = [getchildren(m)[i] for i = start:stop]
        πs = [getprobability(lshaped.structuredModel)[i] for i = start:stop]
        @spawnat w init_subworker(lshaped.subworkers[w-1],m,submodels,πs,collect(start:stop))
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
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    return c⋅x + sum([subprob.π*subprob(x) for subprob in lshaped.subProblems])
end

@implement_traitfn IsParallel function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objidx = [v.col for v in lshaped.structuredModel.obj.aff.vars]
    return c⋅x[objidx] + sum(fetch.([@spawnat w calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
end

# ------------------------------------------------------------------------ #

# ======================================================================== #
