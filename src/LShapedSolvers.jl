module LShapedSolvers

using TraitDispatch
using Parameters
using JuMP
using StochasticPrograms
using MathProgBase
using RecipesBase
using ProgressMeter
using TimerOutputs

import Base: show, put!, wait, isready, take!, fetch
import StochasticPrograms: StructuredModel, optimsolver, optimize_structured!, fill_solution!
importall MathProgBase.SolverInterface

using Base.Order: Ordering, ReverseOrdering, Reverse
using DataStructures: PriorityQueue,enqueue!,dequeue!

export
    LShapedSolver,
    Crash,
    StructuredModel,
    optimsolver,
    optimize_structured!,
    fill_solution!,
    get_solution,
    get_objective_value

# Include files
include("LQSolver.jl")
include("subproblem.jl")
include("hyperplane.jl")
include("AbstractLShaped.jl")
include("regularization.jl")
include("parallel.jl")
include("solvers/LShaped.jl")
include("solvers/DLShaped.jl")
include("solvers/Regularized.jl")
include("solvers/DRegularized.jl")
include("solvers/TrustRegion.jl")
include("solvers/DTrustRegion.jl")
include("solvers/LevelSet.jl")
include("solvers/DLevelSet.jl")
include("crash.jl")
include("spinterface.jl")

end # module
