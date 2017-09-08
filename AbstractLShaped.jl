abstract type AbstractLShapedSolver end

nscenarios(lshaped::AbstractLShapedSolver) = lshaped.nscenarios

function Base.show(io::IO, lshaped::AbstractLShapedSolver)
    print(io,"LShapedSolver")
end

function Base.show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end

function updateSolution!(lshaped::AbstractLShapedSolver)
    lshaped.x = lshaped.masterSolver.x[1:lshaped.structuredModel.numCols]
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

function prepareMaster!(lshaped::AbstractLShapedSolver,n::Integer)
    lshaped.masterSolver = LQSolver(lshaped.structuredModel)

    for i = 1:n
        addvar!(lshaped.masterSolver.model,-1e19,Inf,1.0)
    end
    append!(lshaped.masterSolver.x,zeros(n))
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

# TRAITS #
# ================== #

# IsRegularized -> Algorithm uses the regularized decomposition method of Ruszczyński
# ------------------------------------------------------------
@traitdef IsRegularized{LS}

# HasTrustRegion -> Algorithm uses the trust-region method of Linderoth/Wright
# ------------------------------------------------------------
@traitdef HasTrustRegion{LS}


# UsesLocalization -> Algorithm uses some localization method, applies to both IsRegularized and HasTrustRegion
@traitdef UsesLocalization{LS}
useslocalization(LS) = (istrait(IsRegularized{LS}) || istrait(HasTrustRegion{LS}))
@traitimpl UsesLocalization{LS} <- useslocalization(LS)

@traitfn function checkOptimality{LS <: AbstractLShapedSolver; UsesLocalization{LS}}(lshaped::LS)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    θ = c⋅lshaped.x + sum(lshaped.θs[lshaped.ready])

    if abs(θ - lshaped.Q̃) <= lshaped.τ*(1+abs(lshaped.Q̃))
        return true
    else
        return false
    end
end

@traitfn function checkOptimality{LS <: AbstractLShapedSolver; !UsesLocalization{LS}}(lshaped::LS)
    Q = sum(lshaped.subObjectives)
    θ = sum(lshaped.θs)
    return abs(θ-Q) <= lshaped.τ*(1+abs(θ))
end

@traitfn function queueViolated!{LS <: AbstractLShapedSolver; UsesLocalization{LS}}(lshaped::LS)
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

# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@traitdef IsParallel{LS}

@traitfn function init{LS <: AbstractLShapedSolver; !IsParallel{LS}}(lshaped::LS)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)
    lshaped.nscenarios = n

    prepareMaster!(lshaped,n)
    lshaped.x = similar(m.colVal)
    lshaped.θs = fill(-Inf,n)
    lshaped.obj = Inf
    lshaped.ready = falses(n)

    lshaped.subProblems = Vector{SubProblem}(n)
    π = getprobability(lshaped.structuredModel)
    for i = 1:n
        lshaped.subProblems[i] = SubProblem(getchildren(m)[i],m,i,π[i])
    end
    lshaped.subObjectives = zeros(n)

    lshaped.cuts = Vector{AbstractHyperplane}()
    lshaped.nOptimalityCuts = 0
    lshaped.nFeasibilityCuts = 0

    lshaped.τ = 1e-6
end

@traitfn function calculateObjective{LS <: AbstractLShapedSolver; !IsParallel{LS}}(lshaped::LS,x::AbstractVector)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    return c⋅x + sum([subprob.π*subprob(x) for subprob in lshaped.subProblems])
end
