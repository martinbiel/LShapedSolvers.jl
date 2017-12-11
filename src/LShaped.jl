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
    LPSolver,
    get_solution,
    get_objective_value

JuMPModel = JuMP.Model
JuMPVariable = JuMP.Variable

# Include files
include("LPSolver.jl")
include("subproblem.jl")
include("AbstractLShaped.jl")
include("hyperplane.jl")
include("solvers/LShapedSolver.jl")
#include("solvers/RegularizedLShapedSolver.jl")
#include("solvers/TrustRegionLShapedSolver.jl")

end # module
