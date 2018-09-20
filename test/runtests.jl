using LShapedSolvers
using Base.Test
using JuMP
using StochasticPrograms
using Gurobi
using GLPKMathProgInterface

logging(DevNull, kind=:warn)

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
lsolvers = [(LShapedSolver(:ls,reference_solver,log=false),"L-Shaped"),
            (LShapedSolver(:rd,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"RD L-Shaped"),
            (LShapedSolver(:rd,reference_solver,crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
            (LShapedSolver(:tr,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,reference_solver,log=false),"Leveled L-Shaped"),
            (LShapedSolver(:lv,reference_solver,log=false,linearize=true),"Linearized Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")

info("Test problems loaded. Starting test sequence.")
@testset "$lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

@testset "Bundled $lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    add_params!(lsolver,bundle=2)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

info("Loading infeasible...")
include("infeasible.jl")
lsolvers = [(LShapedSolver(:ls,reference_solver,log=false),"L-Shaped"),
            (LShapedSolver(:tr,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped")]
@testset "$lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    @test solve(sp,solver=lsolver) == :Infeasible
    add_params!(lsolver,checkfeas=true)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end
@testset "Bundled $lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    add_params!(lsolver,checkfeas=false,bundle=2)
    @test solve(sp,solver=lsolver) == :Infeasible
    add_params!(lsolver,checkfeas=true,bundle=2)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

info("Starting distributed tests...")

include("/usr/share/julia/test/testenv.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags run_dtests.jl`

if !success(pipeline(cmd; stdout=STDOUT, stderr=STDERR)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
