module LShaped

using TraitDispatch
using JuMP
using StructJuMP
using Clp
using Gurobi
using MathProgBase

import Base: show, Order.Ordering, Order.Reverse

importall MathProgBase.SolverInterface
import DataStructures: PriorityQueue,enqueue!,dequeue!

export
    LShapedSolver,
    RegularizedLShapedSolver,
    TrustRegionLShapedSolver,
    AsynchronousLShapedSolver,
    updateSubProblem!,
    LPProblem,
    LPSolver,
    loadStandardForm!,
    addRows!,
    addCols!,
    status,
    updateSolution

JuMPModel = JuMP.Model
JuMPVariable = JuMP.Variable

# Include files
include("LPProblem.jl")
include("LPSolver.jl")
include("subproblem.jl")
include("AbstractLShaped.jl")
include("hyperplane.jl")
include("solvers/LShapedSolver.jl")
#include("solvers/RegularizedLShapedSolver.jl")
#include("solvers/TrustRegionLShapedSolver.jl")

end # module
