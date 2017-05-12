module LShaped

# Import functions for overloading
import Base.show

# Export only the useful functions
# export function2

# Packages
using JuMP
using StructJuMP

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

typealias JuMPModel JuMP.Model

# Include files
include("LPSolver.jl")
include("Simplex.jl")
include("LShapedSolver.jl")

end # module
