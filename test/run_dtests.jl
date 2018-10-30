using Test
using LinearAlgebra
using Distributed
include(joinpath(Sys.BINDIR, "..", "share", "julia", "test", "testenv.jl"))
addprocs_with_testenv(3)
@test nworkers() == 3

@everywhere using Logging
for w in workers()
    # Do not log on worker nodes
    remotecall(()->global_logger(NullLogger()),w)
end

@everywhere using StochasticPrograms
using LShapedSolvers
using JuMP
using GLPKMathProgInterface
using Ipopt

τ = 1e-5
reference_solver = GLPKSolverLP()
qp_solver = IpoptSolver(print_level=0)
dlsolvers = [(LShapedSolver(:dls,reference_solver,log=false),"L-Shaped"),
             (LShapedSolver(:drd,qp_solver,subsolver=reference_solver,crash=Crash.EVP(),autotune=true,log=false),"RD L-Shaped"),
             (LShapedSolver(:drd,reference_solver,crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
             (LShapedSolver(:dtr,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
             (LShapedSolver(:dlv,reference_solver,projectionsolver=qp_solver,log=false),"Leveled L-Shaped"),
             (LShapedSolver(:dlv,reference_solver,log=false,linearize=true),"Linearized Leveled L-Shaped")]
lsolvers = [(LShapedSolver(:ls,reference_solver,log=false),"L-Shaped"),
            (LShapedSolver(:rd,qp_solver,subsolver=reference_solver,crash=Crash.EVP(),autotune=true,log=false),"RD L-Shaped"),
            (LShapedSolver(:rd,reference_solver,crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
            (LShapedSolver(:tr,reference_solver,crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,reference_solver,projectionsolver=qp_solver,log=false),"Leveled L-Shaped"),
            (LShapedSolver(:lv,reference_solver,log=false,linearize=true),"Linearized Leveled L-Shaped")]
problems = Vector{Tuple{<:StochasticProgram,String}}()
@info "Loading test problems..."
@info "Loading simple..."
include("simple.jl")
@info "Loading farmer..."
include("farmer.jl")
@info "Loading infeasible..."
include("infeasible.jl")

@testset "Distributed solvers" begin
    @testset "Simple problems" begin
        @info "Test problems loaded. Starting test sequence."
        @testset "Distributed $lsname Solver with Distributed Data: $name" for (lsolver,lsname) in dlsolvers, (sp,name) in problems
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            optimize!(sp, solver=lsolver)
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Distributed Bundled $lsname Solver: $name" for (lsolver,lsname) in dlsolvers, (sp,name) in problems
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            add_params!(lsolver, bundle=2)
            optimize!(sp, solver=lsolver)
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Distributed $lsname Solver: $name" for (lsolver,lsname) in dlsolvers, (sp,name) in problems
            sp_nondist = copy(sp, procs = [1])
            optimize!(sp_nondist, solver=reference_solver)
            x̄ = optimal_decision(sp_nondist)
            Q̄ = optimal_value(sp_nondist)
            optimize!(sp_nondist, solver=lsolver)
            @test abs(optimal_value(sp_nondist) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp_nondist) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "$lsname Solver with Distributed Data: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            with_logger(NullLogger()) do
                optimize!(sp, solver=lsolver)
            end
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
    end
    @testset "Infeasible problem" begin
        @testset "$lsname Solver: Feasibility cuts" for (lsolver,lsname) in dlsolvers
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
        @testset "Bundled $lsname Solver: Feasibility cuts" for (lsolver,lsname) in dlsolvers
            optimize!(infeasible, solver=reference_solver)
            x̄ = optimal_decision(infeasible)
            Q̄ = optimal_value(infeasible)
            add_params!(lsolver, checkfeas=false,bundle=2)
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
