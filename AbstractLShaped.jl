abstract type AbstractLShapedSolver end

nscenarios(lshaped::AbstractLShapedSolver) = lshaped.nscenarios

function Base.show(io::IO, lshaped::AbstractLShapedSolver)
    print(io,"LShapedSolver")
end

function Base.show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end

function updateSolution!(lshaped::AbstractLShapedSolver)
    lshaped.x = lshaped.masterModel.colVal[1:lshaped.structuredModel.numCols]
end

function updateObjectiveValue!(lshaped::AbstractLShapedSolver)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
    lshaped.obj = c⋅lshaped.x + sum(lshaped.subObjectives)
end

function updateStructuredModel!(lshaped::AbstractLShapedSolver)
    c = JuMP.prepAffObjective(lshaped.structuredModel)
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
    lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
    @constraint(lshaped.masterModel,thetabound[i = 1:n],θ[i] >= -1e19)
    lshaped.ready = falses(n)

    updateObjective!(lshaped)

    p = LPProblem(lshaped.masterModel)
    lshaped.masterProblem = p
    lshaped.masterSolver = LPSolver(p)
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

# IsRegularized -> Algorithm uses regularized decomposition
# ------------------------------------------------------------
@traitdef IsRegularized{LS}

@traitfn function updateObjective!{LS <: AbstractLShapedSolver; !IsRegularized{LS}}(lshaped::LS)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objidx = [v.col for v in lshaped.structuredModel.obj.aff.vars]
    x = [Variable(lshaped.masterModel,i) for i in 1:(lshaped.structuredModel.numCols)]

    @objective(lshaped.masterModel,Min,sum(c.*x[objidx]) + sum(lshaped.θs))
end

@traitfn function checkOptimality{LS <: AbstractLShapedSolver; !IsRegularized{LS}}(lshaped::LS)
    Q = sum(lshaped.subObjectives)
    θ = sum(getvalue(lshaped.θs))
    return abs(θ-Q) <= lshaped.τ*(1+abs(θ))
end

# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@traitdef IsParallel{LS}

@traitfn function init{LS <: AbstractLShapedSolver; !IsParallel{LS}}(lshaped::LS)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)

    extractMaster!(lshaped,m)
    prepareMaster!(lshaped,n)
    lshaped.x = m.colVal
    lshaped.obj = Inf

    lshaped.nscenarios = n
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
    return c⋅x + sum([subprob.π*subprob(x) for subprob in lshaped.subProblems])
end
