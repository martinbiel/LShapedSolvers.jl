using LShapedSolvers
using Base.Test
using JuMP
using StochasticPrograms
using Gurobi

solver = GurobiSolver(OutputFlag=0)
lsolvers = [(LShapedSolver(:ls,solver,log=false),"L-Shaped"),
            (LShapedSolver(:rd,solver,crash=Crash.EVP(),σ=60.0,σ̲=10.0,σ̅=200.0,log=false),"RD L-Shaped"),
            (LShapedSolver(:tr,solver,crash=Crash.EVP(),Δ=50.0,Δ̅=100.0,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,solver,crash=Crash.EVP(),λ=0.95,log=false),"Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")

info("Test problems loaded. Starting test sequence.")
@testset "$lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    @test norm(optimal_decision(sp) - x̄) <= 1e-2
    @test abs(optimal_value(sp) - Q̄) <= 1e-2
end

info("Starting distributed tests...")

include("/opt/julia-0.6/share/julia/test/testenv.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags run_dtests.jl`

if !success(pipeline(cmd; stdout=STDOUT, stderr=STDERR)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
