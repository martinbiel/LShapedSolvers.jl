mutable struct SubProblem
    model::JuMPModel
    id::Integer
    π::Float64
    problem::LPProblem
    solver::LPSolver
    parent # Parent LShaped

    hl
    hli
    hu
    hui
    masterTerms

    function SubProblem(m::JuMPModel,id::Integer,π::Float64)
        subprob = new(m,id,π)

        p = LPProblem(m)
        subprob.problem = p
        subprob.solver = LPSolver(p)
        subprob.hl = copy(p.l)
        subprob.hli = find(!isinf,subprob.hl)
        subprob.hu = copy(p.u)
        subprob.hui = find(!isinf,subprob.hu)

        return subprob
    end
end

mutable struct LShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterProblem::LPProblem
    masterSolver::LPSolver

    numScenarios::Integer
    subProblems::Vector{SubProblem}

    # Cuts
    θs
    numOptimalityCuts::Integer
    numFeasibilityCuts::Integer

    status::Symbol
    τ::Float64

    function LShapedSolver(m::JuMPModel)
        @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
        lshaped = new(m)

        n = num_scenarios(m)
        lshaped.numScenarios = n

        lshaped.masterModel = extractMaster!(m)
        lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
        @constraint(lshaped.masterModel,thetabound[i = 1:n],θ[i] >= -1e19)

        p = LPProblem(lshaped.masterModel)
        lshaped.masterProblem = p
        lshaped.masterSolver = LPSolver(p)

        prepareMaster!(lshaped)

        lshaped.subProblems = Vector{SubProblem}()
        π = getprobability(lshaped.structuredModel)
        for i = 1:n
            subprob = SubProblem(lshaped,getchildren(m)[i],i,π[i])
            push!(lshaped.subProblems,subprob)
        end

        lshaped.numOptimalityCuts = 0
        lshaped.numFeasibilityCuts = 0

        lshaped.τ = 1e-6

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

    return master
end

function prepareMaster!(lshaped::LShapedSolver)
    obj = lshaped.masterModel.obj.aff

    for i = 1:lshaped.numScenarios
        push!(obj.vars,lshaped.θs[i])
        push!(obj.coeffs,1)
        addToObjective!(lshaped.masterProblem,lshaped.masterModel,lshaped.θs[i],1)
    end
end

function updateMasterSolution!(lshaped::LShapedSolver)
    @assert status(lshaped.masterSolver) == :Optimal "Should not update unoptimized result"

    for i in 1:lshaped.structuredModel.numCols
        lshaped.structuredModel.colVal[i] = lshaped.masterModel.colVal[i]
    end
end

function addCut!(lshaped::LShapedSolver,subprob::SubProblem)

    m = lshaped.masterModel
    cutIdx = m.numCols-lshaped.numScenarios

    substatus = status(subprob.solver)

    if substatus == :Optimal

        x = lshaped.structuredModel.colVal

        E,e = getOptimalityCut(subprob)
        w = e-E⋅x
        θ = getvalue(lshaped.θs[subprob.id])
        τ = lshaped.τ

        if abs((w-θ)/θ) <= τ
            # Optimal with respect to this subproblem
            println("θ",subprob.id,": ",getvalue(lshaped.θs[subprob.id]))
            println("w",subprob.id,": ", w)
            println("Optimal with respect to subproblem ",subprob.id)
            return false
        end

        # Add optimality cut
        @constraint(m,sum(E[i]*Variable(m,i)
                          for i = 1:cutIdx) + Variable(m,cutIdx+subprob.id) >= e)
        addRows!(lshaped.masterProblem,m)
        lshaped.numOptimalityCuts += 1
        println("θ",subprob.id,": ",getvalue(lshaped.θs[subprob.id]))
        println("w",subprob.id,": ", w)
        if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
            println("Added Optimality Cut")
        else
            println("Added Optimality Cut: ", lshaped.masterModel.linconstr[end])
        end
        return true
    elseif substatus == :Infeasible
        D,d = getFeasibilityCut(subprob)

        # Scale to avoid numerical issues
        scaling = abs(d)
        if scaling == 0
            scaling = maximum(D)
        end

        D = D/scaling

        # Add feasibility cut
        @constraint(m,sum(D[i]*Variable(m,i)
                           for i = 1:cutIdx) >= sign(d))
        addRows!(lshaped.masterProblem,m)
        lshaped.numFeasibilityCuts += 1
        println("Subproblem ",subprob.id, " is infeasible")
        if length(lshaped.masterModel.linconstr[end].terms.coeffs) > 10
            println("Added Feasibility Cut")
        else
            println("Added Feasibility Cut: ", lshaped.masterModel.linconstr[end])
        end
        return true
    else
        warn("Subproblem ",subprob,id," was not solved")
    end

    return false
end

function (lshaped::LShapedSolver)()
    println("Starting L-Shaped procedure\n")
    println("======================")
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    updateSolution(lshaped.masterSolver,lshaped.masterModel)
    lshaped.status = status(lshaped.masterSolver)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateMasterSolution!(lshaped)

    # Initial update of sub problems
    map(updateSubProblem!,lshaped.subProblems)

    addedCut = false

    println("Main loop")
    println("======================")

    while true
        # Solve sub problems
        for subprob in lshaped.subProblems
            println("Solving subproblem: ",subprob.id)
            subprob.solver()
            updateSolution(subprob.solver,subprob.model)
            lshaped.status = status(subprob.solver)
            if lshaped.status == :Unbounded
                println("Subproblem ",subprob.id," is unbounded, aborting procedure.")
                println("======================")
                return
            end
            addedCut |= addCut!(lshaped,subprob)
        end

        if !addedCut
            # Optimal
            lshaped.status = :Optimal
            println("Optimal!")
            println("======================")
            break
        end

        # Resolve master
        lshaped.masterSolver()
        updateSolution(lshaped.masterSolver,lshaped.masterModel)
        lshaped.status = status(lshaped.masterSolver)
        if lshaped.status == :Infeasible
            println("Master is infeasible, aborting procedure.")
            println("======================")
            return
        end

        # Update master solution
        updateMasterSolution!(lshaped)

        # Update subproblems
        map(updateSubProblem!,lshaped.subProblems)

        # Reset
        addedCut = false
    end
end

function SubProblem(parent::LShapedSolver,m::JuMPModel,id::Integer,π::Float64)
    subprob = SubProblem(m,id,π)

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

    numCols = subprob.problem.numCols

    for (i,rhs) in rhsupdate
        constr = subprob.model.linconstr[i]
        subprob.problem.l[numCols+i] = -(constr.ub + rhs)
        subprob.problem.u[numCols+i] = -(constr.lb + rhs)
    end
end

function getOptimalityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Optimal
    λ = subprob.solver.λ
    v = subprob.solver.v
    w = subprob.solver.w
    hl = subprob.hl
    hu = subprob.hu
    π = subprob.π
    E = zeros(subprob.parent.structuredModel.numCols)

    e = π*v[subprob.hli]⋅hl[subprob.hli] + π*w[subprob.hui]⋅hu[subprob.hui]

    for (i,x,coeff) in subprob.masterTerms
        E[x.col] += π*λ[i]*(-coeff)
    end

    return E, e
end

function getFeasibilityCut(subprob::SubProblem)
    @assert status(subprob.solver) == :Infeasible
    λ = subprob.solver.λ
    v = subprob.solver.v
    w = subprob.solver.w
    hl = subprob.hl
    hu = subprob.hu
    D = zeros(subprob.parent.structuredModel.numCols)

    d = v[subprob.hli]⋅hl[subprob.hli] + w[subprob.hui]⋅hu[subprob.hui]

    for (i,x,coeff) in subprob.masterTerms
        D[x.col] += λ[i]*(-coeff)
    end

    return D, d
end
