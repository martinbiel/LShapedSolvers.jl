mutable struct LShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterProblem::LPProblem
    masterSolver::LPSolver
    x::AbstractVector
    obj::Real

    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Cuts
    θs
    ready
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer
    cuts::Vector{AbstractHyperplane}

    status::Symbol
    τ::Float64

    function LShapedSolver(m::JuMPModel)
        lshaped = new(m)

        init(lshaped)
        lshaped.cuts = Vector{AbstractHyperplane}()

        return lshaped
    end
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
    updateSolution!(lshaped)

    println("Main loop")
    println("======================")

    while true
        # Resolve all subproblems at the current optimal solution
        resolveSubproblems!(lshaped)

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            updateStructuredModel!(lshaped)
            println("Optimal!")
            println("======================")
            break
        end

        # Resolve master
        println("Solving master problem")
        lshaped.masterSolver()
        updateSolution(lshaped.masterSolver,lshaped.masterModel)
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
