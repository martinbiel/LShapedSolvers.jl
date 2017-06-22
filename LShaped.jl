module LShaped

# Import functions for overloading
import Base.show

# Export only the useful functions
# export function2

# Packages
using JuMP
using StructJuMP
using Clp
using Gurobi
using MathProgBase.linprog

export
    LShapedSolver,
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
include("LPSolver.jl")
#include("Simplex.jl")
include("LShapedSolver.jl")

end # module
