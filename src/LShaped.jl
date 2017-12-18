module LShaped

using TraitDispatch
using Parameters
using JuMP
using StructJuMP
using MathProgBase
using RecipesBase
using TimerOutputs
using Plots.font

importall MathProgBase.SolverInterface

using Base.Order: Ordering, ReverseOrdering, Reverse
using DataStructures: PriorityQueue,enqueue!,dequeue!

export
    LShapedSolver,
    RegularizedLShapedSolver,
    TrustRegionLShapedSolver,
    PLShapedSolver,
    LPSolver,
    get_solution,
    get_objective_value

JuMPModel = JuMP.Model
JuMPVariable = JuMP.Variable

# Include files
include("LPSolver.jl")
include("subproblem.jl")
include("hyperplane.jl")
include("AbstractLShaped.jl")
include("localization.jl")
include("parallel.jl")
include("solvers/LShapedSolver.jl")
include("solvers/PLShapedSolver.jl")
include("solvers/RegularizedLShapedSolver.jl")
include("solvers/TrustRegionLShapedSolver.jl")

end # module
