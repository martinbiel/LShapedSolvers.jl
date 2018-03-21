module LShapedSolvers

using TraitDispatch
using Parameters
using JuMP
using StochasticPrograms
using MathProgBase
using RecipesBase
using TimerOutputs

import Base: show, put!, wait, isready, take!, fetch
import StochasticPrograms: StructuredModel, optimize_structured!, fill_solution!
importall MathProgBase.SolverInterface

using Base.Order: Ordering, ReverseOrdering, Reverse
using DataStructures: PriorityQueue,enqueue!,dequeue!

export
    LShapedSolver,
    StructuredModel,
    optimize_structured!,
    fill_solution!,
    get_solution,
    get_objective_value

# Include files
include("LPSolver.jl")
include("subproblem.jl")
include("hyperplane.jl")
include("AbstractLShaped.jl")
include("regularization.jl")
include("parallel.jl")
include("solvers/LShaped.jl")
include("solvers/ALShaped.jl")
include("solvers/Regularized.jl")
include("solvers/TrustRegion.jl")
include("solvers/ATrustRegion.jl")
include("solvers/LevelSet.jl")
include("spinterface.jl")

end # module
