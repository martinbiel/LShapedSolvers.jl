using Base.Test
include("/opt/julia-0.6/share/julia/test/testenv.jl")
addprocs_with_testenv(3)
@test nworkers() == 3

using LShapedSolvers
using JuMP
using StochasticPrograms
using Gurobi

solver = GurobiSolver(OutputFlag=0)
lsolvers = [(LShapedSolver(:dls,solver,κ=1.0,log=false),"L-Shaped"),(LShapedSolver(:drd,solver,σ=60.0,σ̲=10.0,σ̅=200.0,κ=1.0,log=false),"RD L-Shaped"),(LShapedSolver(:dtr,solver,Δ=50.0,Δ̅=100.0,κ=1.0,log=false),"TR L-Shaped"),(LShapedSolver(:dlv,solver,λ=1.0,κ=1.0,log=false),"Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")

info("Test problems loaded. Starting test sequence.")
@testset "Distributed $lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    @test norm(sp.colVal - x̄) <= 1e-2
    @test abs(sp.objVal - Q̄) <= 1e-2
end
