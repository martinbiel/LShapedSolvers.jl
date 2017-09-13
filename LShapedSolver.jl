mutable struct LShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    # Master
    masterSolver::AbstractLQSolver
    x::AbstractVector
    obj::Real
    obj_hist::AbstractVector

    # Subproblems
    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Cuts
    θs::AbstractVector
    cuts::Vector{AbstractHyperplane}
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer

    status::Symbol
    τ::Float64

    function LShapedSolver(m::JuMPModel,x₀::AbstractVector)
        lshaped = new(m)

        if length(x₀) != m.numCols
            throw(ArgumentError(string("Incorrect length of starting guess, has ",length(x₀)," should be ",m.numCols)))
        end

        lshaped.x = x₀
        init(lshaped)

        return lshaped
    end
end
LShapedSolver(m::JuMPModel) = LShapedSolver(m,rand(m.numCols))

function (lshaped::LShapedSolver)()
    println("Starting L-Shaped procedure")
    println("======================")

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
