using LShapedSolvers
using Test
using Logging
using Distributed
using JuMP
using StochasticPrograms
using GLPKMathProgInterface

τ = 1e-5
reference_solver = GLPKSolverLP()
lsolvers = [(LShapedSolver(:ls,reference_solver,log=false),"L-Shaped"),
            (LShapedSolver(:rd,reference_solver,crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
            (LShapedSolver(:tr,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,reference_solver,log=false,linearize=true),"Linearized Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String}}()
@info "Loading test problems..."
@info "Loading simple..."
include("simple.jl")
@info "Loading farmer..."
include("farmer.jl")
@info "Loading infeasible..."
include("infeasible.jl")
@info "Test problems loaded. Starting test sequence."

@testset "Sequential solvers" begin
    @testset "Simple problems" begin
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
    end
    @testset "Infeasible problem" begin
        @testset "$lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
            solve(infeasible,solver=reference_solver)
            x̄ = optimal_decision(infeasible)
            Q̄ = optimal_value(infeasible)
            with_logger(NullLogger()) do
                @test solve(infeasible,solver=lsolver) == :Infeasible
            end
            add_params!(lsolver,checkfeas=true)
            solve(infeasible,solver=lsolver)
            @test abs(optimal_value(infeasible) - Q̄)/(1e-10+abs(Q̄)) <= τ
        end
        @testset "Bundled $lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
            solve(infeasible,solver=reference_solver)
            x̄ = optimal_decision(infeasible)
            Q̄ = optimal_value(infeasible)
            add_params!(lsolver,checkfeas=false,bundle=2)
            with_logger(NullLogger()) do
                @test solve(infeasible,solver=lsolver) == :Infeasible
            end
            add_params!(lsolver,checkfeas=true,bundle=2)
            solve(infeasible,solver=lsolver)
            @test abs(optimal_value(infeasible) - Q̄)/(1e-10+abs(Q̄)) <= τ
        end
    end
end

@info "Starting distributed tests..."

include(joinpath(Sys.BINDIR, "..", "share", "julia", "test", "testenv.jl"))
disttestfile = joinpath(@__DIR__, "run_dtests.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags $disttestfile`

if !success(pipeline(cmd; stdout=stdout, stderr=stderr)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
