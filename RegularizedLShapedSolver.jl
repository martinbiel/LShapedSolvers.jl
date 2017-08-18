mutable struct RegularizedLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterSolver::AbstractMathProgSolver
    gurobienv::Gurobi.Env

    subProblems::Vector{SubProblem}

    # Regularizer
    a::Vector{Float64}
    Qa::Float64

    # Cuts
    θs
    ready
    numOptimalityCuts::Integer
    numFeasibilityCuts::Integer

    status::Symbol
    τ::Float64

    function RegularizedLShapedSolver(m::JuMPModel,a::Vector{Float64})
        lshaped = new(m)

        if length(a) != m.numCols
            throw(ArgumentError(string("Incorrect length of regularizer, has ",length(a)," should be ",m.numCols)))
        end
        lshaped.a = a

        init(lshaped)

        return lshaped
    end
end

@traitimpl IsRegularized{RegularizedLShapedSolver}

RegularizedLShapedSolver(m::JuMPModel,a::AbstractVector) = RegularizedLShapedSolver(m,convert(Vector{Float64},a))

function Base.show(io::IO, lshaped::RegularizedLShapedSolver)
    print(io,"RegularizedLShapedSolver")
end

@traitfn function prepareMaster!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS,n)
    lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
    lshaped.ready = falses(n)

    updateObjective!(lshaped)

    lshaped.gurobienv = Gurobi.Env()
    setparam!(lshaped.gurobienv,"OutputFlag",0)
    lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
    setsolver(lshaped.masterModel,lshaped.masterSolver)
end

@traitfn function updateObjective!{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objinds = [v.col for v in lshaped.structuredModel.obj.aff.vars]
    x = [Variable(lshaped.masterModel,i) for i in 1:(lshaped.structuredModel.numCols)]

    @objective(lshaped.masterModel,Min,sum(c.*x[objinds]) + sum(lshaped.θs[lshaped.ready]) + 0.5*sum((x-lshaped.a).*(x-lshaped.a)))
end

# Regularized implementation
@traitfn function checkOptimality{LS <: AbstractLShapedSolver; IsRegularized{LS}}(lshaped::LS)
    obj = lshaped.structuredModel.objVal
    obj *= lshaped.structuredModel.objSense == :Min ? 1 : -1

    @show obj
    @show lshaped.Qa

    if abs((obj - lshaped.Qa)/lshaped.Qa) <= lshaped.τ
        return true
    else
        return false
    end
end

function (lshaped::RegularizedLShapedSolver)()
    println("Starting L-Shaped procedure\n")
    println("======================")
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.status = solve(lshaped.masterModel)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateMasterSolution!(lshaped)

    # Initial update of sub problems
    updateSubProblems!(lshaped.subProblems,lshaped.structuredModel.colVal)

    addedCut = false

    println("Main loop")
    println("======================")

    while true
        # Solve sub problems
        for subprob in lshaped.subProblems
            println("Solving subproblem: ",subprob.id)
            cut = subprob()
            if !proper(cut)
                println("Subproblem ",subprob.id," is unbounded, aborting procedure.")
                println("======================")
                return
            end
            addedCut |= addCut!(lshaped,cut)
        end

        obj = getobjectivevalue(lshaped.structuredModel)
        obj *= lshaped.structuredModel.objSense == :Min ? 1 : -1
        if !addedCut || (obj - lshaped.Qa) <= lshaped.τ
            lshaped.a = lshaped.structuredModel.colVal
            lshaped.Qa = obj
        end

        updateObjective!(lshaped)

        # Resolve master
        println("Solving master problem")
        lshaped.status = solve(lshaped.masterModel)
        if lshaped.status != :Optimal
            setparam!(lshaped.gurobienv,"Presolve",2)
            setparam!(lshaped.gurobienv,"BarHomogeneous",1)
            lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
            setsolver(lshaped.masterModel,lshaped.masterSolver)
            lshaped.status = solve(lshaped.masterModel)
            if lshaped.status == :Optimal
                setparam!(lshaped.gurobienv,"Presolve",-1)
                setparam!(lshaped.gurobienv,"BarHomogeneous",-1)
                lshaped.masterSolver = GurobiSolver(lshaped.gurobienv)
                setsolver(lshaped.masterModel,lshaped.masterSolver)
            else
                if lshaped.status == :Infeasible
                    println("Master is infeasible, aborting procedure.")
                else
                    println("Master could not be solved, aborting procedure")
                end
                println("======================")
                return
            end
        end
        # Update master solution
        updateMasterSolution!(lshaped)

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            println("Optimal!")
            println("======================")
            break
        end

        # Update subproblems
        updateSubProblems!(lshaped.subProblems,lshaped.structuredModel.colVal)

        # Reset
        addedCut = false
    end
end
