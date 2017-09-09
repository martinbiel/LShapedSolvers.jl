mutable struct LShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterSolver::AbstractLQSolver
    x::AbstractVector
    obj::Real
    obj_hist::AbstractVector

    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Cuts
    θs::AbstractVector
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer
    cuts::Vector{AbstractHyperplane}

    status::Symbol
    τ::Float64

    function LShapedSolver(m::JuMPModel,x₀::AbstractVector)
        lshaped = new(m)

        lshaped.x = x₀
        init(lshaped)

        return lshaped
    end
end
LShapedSolver(m::JuMPModel) = LShapedSolver(m,rand(m.numCols))

function (lshaped::LShapedSolver)()
    println("Starting L-Shaped procedure")
    println("======================")
    println("Initial solve of subproblems at initial guess")
    updateSubProblems!(lshaped.subProblems,lshaped.x)
    map(s->addCut!(lshaped,s(),lshaped.x),lshaped.subProblems)
    push!(lshaped.obj_hist,sum(lshaped.subObjectives))
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    lshaped.status = status(lshaped.masterSolver)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateSolution!(lshaped)

    println("Main loop")
    println("======================")

    while true
        # Resolve all subproblems at the current optimal solution
        resolveSubproblems!(lshaped)
        updateObjectiveValue!(lshaped)
        push!(lshaped.obj_hist,lshaped.obj)

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            updateStructuredModel!(lshaped)
            println("Optimal!")
            println("Objective value: ", sum(lshaped.subObjectives))
            println("======================")
            break
        end

        # Resolve master
        println("Solving master problem")
        lshaped.masterSolver()
        lshaped.status = status(lshaped.masterSolver)
        if lshaped.status == :Infeasible
            println("Master is infeasible, aborting procedure.")
            println("======================")
            return
        end
        # Update master solution
        updateSolution!(lshaped)
    end
end
