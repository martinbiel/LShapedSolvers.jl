__precompile__()
module LShapedSolvers

# Standard library
using LinearAlgebra
using SparseArrays
using Distributed
using Printf

# External libraries
using TraitDispatch
using Parameters
using JuMP
using StochasticPrograms
using MathProgBase
using RecipesBase
using ProgressMeter

import Base: show, put!, wait, isready, take!, fetch, zero, +, length
import StochasticPrograms: StructuredModel, internal_solver, optimize_structured!, fill_solution!, solverstr

const MPB = MathProgBase

export
    LShapedSolver,
    Crash,
    StructuredModel,
    add_params!,
    optimsolver,
    optimize_structured!,
    fill_solution!,
    get_decision,
    get_objective_value,
    LShaped,
    DLShaped,
    Regularized,
    DRegularized,
    TrustRegion,
    DTrustRegion,
    LevelSet,
    DLevelSet

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
