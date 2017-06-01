type SubProblem
    model::JuMPModel
    problem::LPProblem
    solver::LPSolver
    parent # Parent LShaped

    h
    masterTerms

    function SubProblem(m::JuMPModel)
        subprob = new(m)

        p = LPProblem(m)
        subprob.problem = p
        subprob.solver = LPSolver(p)
        subprob.h = copy(p.b)

        return subprob
    end
end

type LShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterProblem::LPProblem
    masterSolver::LPSolver

    numScenarios::Int
    subProblems::Vector{SubProblem}

    # Cuts
    θ
    numCuts::Int

    status

    function LShapedSolver(m::JuMPModel)
        @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
        lshaped = new(m)

        lshaped.masterModel = extractMaster!(m)
        lshaped.θ = @variable(lshaped.masterModel,θ)

        p = LPProblem(lshaped.masterModel)
        lshaped.masterProblem = p
        lshaped.masterSolver = LPSolver(p)

        n = num_scenarios(m)
        lshaped.numScenarios = n
        lshaped.subProblems = Vector{SubProblem}()
        for i = 1:n
            subprob = SubProblem(lshaped,getchildren(m)[i])
            push!(lshaped.subProblems,subprob)
        end

        lshaped.numCuts = 0

        return lshaped
    end
end

function Base.show(io::IO, solver::LShapedSolver)
    if get(io, :multiline, false)
        print(io, "LShapedSolver")
    else
        print(io,"LShapedSolver")
    end
end

function Base.show(io::IO, ::MIME"text/plain", solver::LShapedSolver)
    show(io,solver)
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

    # Variable dicts
    master.varDict = Dict{Symbol,Any}()
    for (symb,v) in src.varDict
        master.varDict[symb] = copy(v, master)
    end

    return master
end

function prepareMaster!(lshaped::LShapedSolver)
    obj = lshaped.masterModel.obj.aff

    push!(obj.vars,lshaped.θ)
    push!(obj.coeffs,1)

    addToObjective!(lshaped.masterProblem,lshaped.masterModel,lshaped.θ,1)

    # Let θ be -Inf initially
    setvalue(lshaped.θ,-Inf)
end

function updateMasterSolution!(lshaped::LShapedSolver)
    @assert status(lshaped.masterSolver) == :Optimal "Should not update unoptimized result"

    for i in 1:lshaped.structuredModel.numCols
        lshaped.structuredModel.colVal[i] = lshaped.masterModel.colVal[i]
    end
end

function addCut!(lshaped::LShapedSolver,E::AbstractVector,e::Real)
    m = lshaped.masterModel
    @constraint(lshaped.masterModel,sum(E[i]*Variable(m,i)
                                        for i = 1:(m.numCols-1)) + Variable(m,m.numCols) >= e)
    addRows!(lshaped.masterProblem,m)

    lshaped.numCuts += 1

    return false
end

function (lshaped::LShapedSolver)()
    println("Starting L-Shaped procedure\n")
    π = getprobability(lshaped.structuredModel)
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    updateMasterSolution!(lshaped)

    # Initial update of sub problems
    map(updateSubProblem!,lshaped.subProblems)

    # Can now prepare master for future cuts
    println("Prepare master problem\n")
    prepareMaster!(lshaped)

    while true
        # Solve sub problems
        for (i,subprob) in enumerate(lshaped.subProblems)
            println("Solving subproblem: ",i)
            subprob.solver()
        end

        E = zeros(lshaped.structuredModel.numCols)
        e = 0.0

        for i = 1:lshaped.numScenarios
            Es,es = getCut(lshaped.subProblems[i])
            E += π[i]*Es
            e += π[i]*es
        end

        x = lshaped.structuredModel.colVal
        w = e-E⋅x

        if w <= getvalue(lshaped.θ)+eps()
            # Optimal
            println("======================")
            println("Current θ: ", getvalue(lshaped.θ))
            println("Current w: ", w)
            println("Optimal!")
            println("======================")
            break
        end

        addCut!(lshaped,E,e)

        println("======================")
        println("Current θ: ", getvalue(lshaped.θ))
        println("Current w: ", w)
        println("Added cut: ", lshaped.masterModel.linconstr[end])
        println("======================")

        # Resolve master
        lshaped.masterSolver()
        updateMasterSolution!(lshaped)

        # Update subproblems
        map(updateSubProblem!,lshaped.subProblems)
    end
end

function SubProblem(parent::LShapedSolver,m::JuMPModel)
    subprob = SubProblem(m)

    subprob.parent = parent

    subprob.masterTerms = []
    parseSubProblem!(subprob)

    return subprob
end

function parseSubProblem!(subprob::SubProblem)
    for (i,constr) in enumerate(subprob.model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == subprob.parent.structuredModel
                # var is a first stage variable
                push!(subprob.masterTerms,(i,var,-constr.terms.coeffs[j]))
            end
        end
    end
end

function updateSubProblem!(subprob::SubProblem)
    @assert status(subprob.parent.masterSolver) == :Optimal

    rhsupdate = Dict{Int64,Float64}()
    for (i,x,coeff) in subprob.masterTerms
        if !haskey(rhsupdate,i)
            rhsupdate[i] = 0
        end
        rhsupdate[i] += coeff*getvalue(x)
    end

    for (i,rhs) in rhsupdate
        constr = subprob.model.linconstr[i]
        if constr.lb == constr.ub
            rhs += constr.ub
        elseif constr.lb == -Inf
            rhs += constr.ub
        elseif constr.ub == Inf
            rhs += constr.lb
        else
            error(string("Can only handle equality constraints or one-sided inequality constraints: ", constr))
        end
        subprob.problem.b[i] = rhs
    end
end

function getCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Optimal
    E = zeros(subprob.parent.structuredModel.numCols)
    e = subprob.solver.λ⋅subprob.h
    for (i,x,coeff) in subprob.masterTerms
        E[x.col] += subprob.solver.λ[i]*(-coeff)
    end

    return E, e
end
