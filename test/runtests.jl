using LShapedSolvers
using LinearAlgebra
using Test
using Logging
using JuMP
using StochasticPrograms
using GLPKMathProgInterface
using Ipopt

τ = 1e-5
reference_solver = GLPKSolverLP()
qp_solver = IpoptSolver(print_level=0)
lsolvers = [(LShapedSolver(reference_solver, log=false),"L-Shaped"),
            (LShapedSolver(qp_solver, subsolver=reference_solver, crash=Crash.EVP(), regularization=:rd, autotune=true, log=false),"RD L-Shaped"),
            (LShapedSolver(reference_solver, crash=Crash.EVP(), regularization=:rd, autotune=true, log=false, linearize=true),"Linearized RD L-Shaped"),
            (LShapedSolver(reference_solver, crash=Crash.EVP(), regularization=:tr, autotune=true, log=false),"TR L-Shaped"),
            (LShapedSolver(reference_solver, projectionsolver=qp_solver, regularization=:lv, log=false),"Leveled L-Shaped"),
            (LShapedSolver(reference_solver, regularization=:lv, log=false, linearize=true),"Linearized Leveled L-Shaped")]
problems = Vector{Tuple{<:StochasticProgram,String}}()
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
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            optimize!(sp, solver=lsolver)
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Bundled $lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            add_params!(lsolver, bundle=2)
            optimize!(sp, solver=lsolver)
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
    end
    @testset "Infeasible problem" begin
        @testset "$lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
            optimize!(infeasible, solver=reference_solver)
            x̄ = optimal_decision(infeasible)
            Q̄ = optimal_value(infeasible)
            with_logger(NullLogger()) do
                @test optimize!(infeasible, solver=lsolver) == :Infeasible
            end
            add_params!(lsolver, checkfeas=true)
            optimize!(infeasible, solver=lsolver)
            @test abs(optimal_value(infeasible) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(infeasible) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Bundled $lsname Solver: Feasibility cuts" for (lsolver,lsname) in lsolvers
            optimize!(infeasible, solver=reference_solver)
            x̄ = optimal_decision(infeasible)
            Q̄ = optimal_value(infeasible)
            add_params!(lsolver, checkfeas=false, bundle=2)
            with_logger(NullLogger()) do
                @test optimize!(infeasible, solver=lsolver) == :Infeasible
            end
            add_params!(lsolver, checkfeas=true, bundle=2)
            optimize!(infeasible, solver=lsolver)
            @test abs(optimal_value(infeasible) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(infeasible) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
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
