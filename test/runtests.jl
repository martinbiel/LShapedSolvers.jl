using LShapedSolvers
using Base.Test
using JuMP
using StochasticPrograms
using Gurobi

τ = 1e-5
solver = GurobiSolver(OutputFlag=0)
lsolvers = [(LShapedSolver(:ls,solver,log=false),"L-Shaped"),
            (LShapedSolver(:rd,solver,crash=Crash.EVP(),τ=1e-6,autotune=true,log=false),"RD L-Shaped"),
            (LShapedSolver(:tr,solver,crash=Crash.EVP(),τ=1e-6,autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,solver,crash=Crash.EVP(),log=false),"Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String,Bool}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")
info("Loading day-ahead problems...")
include("dayahead.jl")

info("Test problems loaded. Starting test sequence.")
@testset "$lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name,flatobj) in problems
    solve(sp,solver=solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    if !flatobj
        @test norm(optimal_decision(sp) - x̄) <= τ*(norm(x̄)+1e-10)
    end
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end

info("Starting distributed tests...")

include("/opt/julia-0.6/share/julia/test/testenv.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags run_dtests.jl`

if !success(pipeline(cmd; stdout=STDOUT, stderr=STDERR)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
