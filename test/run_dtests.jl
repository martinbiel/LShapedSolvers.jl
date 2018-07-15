using Base.Test
include("/opt/julia-0.6/share/julia/test/testenv.jl")
addprocs_with_testenv(3)
@test nworkers() == 3

using LShapedSolvers
using JuMP
using StochasticPrograms
using Gurobi

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
dlsolvers = [(LShapedSolver(:dls,GurobiSolver(OutputFlag=0),log=false),"L-Shaped"),
             (LShapedSolver(:drd,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"RD L-Shaped"),
             (LShapedSolver(:dtr,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
             (LShapedSolver(:dlv,GurobiSolver(OutputFlag=0),log=false),"Leveled L-Shaped")]
lsolvers = [(LShapedSolver(:ls,GurobiSolver(OutputFlag=0),log=false),"L-Shaped"),
            (LShapedSolver(:rd,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"RD L-Shaped"),
            (LShapedSolver(:tr,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),log=false),"Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String,Bool}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")
info("Loading day-ahead problems...")
include("dayahead.jl")

info("Test problems loaded. Starting test sequence.")
@testset "Distributed $lsname Solver with Distributed Data: $name" for (lsolver,lsname) in dlsolvers, (sp,name,flatobj) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    if !flatobj
        @test norm(optimal_decision(sp) - x̄) <= τ*(norm(x̄)+1e-10)
    end
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end

@testset "Distributed $lsname Solver: $name" for (lsolver,lsname) in dlsolvers, (sp,name,flatobj) in problems
    sp_nondist = StochasticProgram(first_stage_data(sp),second_stage_data(sp),scenarios(sp),procs=[1])
    transfer_model!(stochastic(sp_nondist),stochastic(sp))
    generate!(sp_nondist)
    solve(sp_nondist,solver=reference_solver)
    x̄ = copy(sp_nondist.colVal)
    Q̄ = copy(sp_nondist.objVal)
    solve(sp_nondist,solver=lsolver)
    if !flatobj
        @test norm(optimal_decision(sp) - x̄) <= τ*(norm(x̄)+1e-10)
    end
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end

@testset "$lsname Solver with Distributed Data: $name" for (lsolver,lsname) in lsolvers, (sp,name,flatobj) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    if !flatobj
        @test norm(optimal_decision(sp) - x̄) <= τ*(norm(x̄)+1e-10)
    end
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end
