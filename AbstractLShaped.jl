abstract type AbstractLShapedSolver end

nscenarios(lshaped::AbstractLShapedSolver) = length(lshaped.subProblems)

function Base.show(io::IO, lshaped::AbstractLShapedSolver)
    print(io,"LShapedSolver")
end

function Base.show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end

function updateMasterSolution!(lshaped::AbstractLShapedSolver)
    for i in 1:lshaped.structuredModel.numCols
        lshaped.structuredModel.colVal[i] = lshaped.masterModel.colVal[i]
    end

    x = lshaped.structuredModel.colVal
    c = JuMP.prepAffObjective(lshaped.structuredModel)

    lshaped.structuredModel.objVal = c⋅x + sum(getvalue(lshaped.θs[lshaped.ready]))
    lshaped.structuredModel.objVal *= lshaped.structuredModel.objSense == :Min ? 1 : -1
end

function extractMaster!(src::JuMPModel)
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

    return master
end

# TRAITS #
# ================== #

# IsRegularized -> Algorithm uses regularized decomposition
# ------------------------------------------------------------
@traitdef IsRegularized{LS}

@traitfn function prepareMaster!{LS <: AbstractLShapedSolver; !IsRegularized{LS}}(lshaped::LS,n)
    lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
    @constraint(lshaped.masterModel,thetabound[i = 1:n],θ[i] >= -1e19)
    lshaped.ready = falses(n)

    updateObjective!(lshaped)

    p = LPProblem(lshaped.masterModel)
    lshaped.masterProblem = p
    lshaped.masterSolver = LPSolver(p)
end

@traitfn function updateObjective!{LS <: AbstractLShapedSolver; !IsRegularized{LS}}(lshaped::LS)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objinds = [v.col for v in lshaped.structuredModel.obj.aff.vars]
    x = [Variable(lshaped.masterModel,i) for i in objinds]

    @objective(lshaped.masterModel,Min,sum(c.*x) + sum(lshaped.θs))
end

@traitfn checkOptimality{LS <: AbstractLShapedSolver; !IsRegularized{LS}}(lshaped::LS) = all([!addCut!(lshaped,subprob()) for subprob in lshaped.subProblems])

# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@traitdef IsParallel{LS}

@traitfn function init{LS <: AbstractLShapedSolver; !IsParallel{LS}}(lshaped::LS)
    m = lshaped.structuredModel
    @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
    n = num_scenarios(m)

    lshaped.masterModel = extractMaster!(m)
    prepareMaster!(lshaped,n)

    lshaped.subProblems = Vector{SubProblem}(n)
    π = getprobability(lshaped.structuredModel)
    for i = 1:n
        lshaped.subProblems[i] = SubProblem(getchildren(m)[i],m,i,π[i])
    end

    lshaped.numOptimalityCuts = 0
    lshaped.numFeasibilityCuts = 0

    lshaped.τ = 1e-6
end
