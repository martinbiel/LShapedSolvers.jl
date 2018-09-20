using Base.Test
include("/usr/share/julia/test/testenv.jl")
addprocs_with_testenv(3)
@test nworkers() == 3

@everywhere using StochasticPrograms
using LShapedSolvers
using JuMP
using Gurobi

@everywhere logging(DevNull, kind=:warn)

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
dlsolvers = [(LShapedSolver(:dls,GurobiSolver(OutputFlag=0),log=false),"L-Shaped"),
             (LShapedSolver(:drd,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
             (LShapedSolver(:dtr,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
             (LShapedSolver(:dlv,GurobiSolver(OutputFlag=0),log=false,linearize=true),"Linearized Leveled L-Shaped")]

lsolvers = [(LShapedSolver(:ls,GurobiSolver(OutputFlag=0),log=false),"L-Shaped"),
            (LShapedSolver(:rd,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false,linearize=true),"Linearized RD L-Shaped"),
            (LShapedSolver(:tr,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped"),
            (LShapedSolver(:lv,GurobiSolver(OutputFlag=0),log=false,linearize=true),"Linearized Leveled L-Shaped")]

problems = Vector{Tuple{JuMP.Model,String}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")

info("Test problems loaded. Starting test sequence.")
@testset "Distributed $lsname Solver with Distributed Data: $name" for (lsolver,lsname) in dlsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

@testset "Distributed Bundled $lsname Solver: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    add_params!(lsolver,bundle=2)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

@testset "Distributed $lsname Solver: $name" for (lsolver,lsname) in dlsolvers, (sp,name) in problems
    sp_nondist = StochasticProgram(first_stage_data(sp),second_stage_data(sp),scenarios(sp),procs=[1])
    transfer_model!(stochastic(sp_nondist),stochastic(sp))
    generate!(sp_nondist)
    solve(sp_nondist,solver=reference_solver)
    x̄ = copy(sp_nondist.colVal)
    Q̄ = copy(sp_nondist.objVal)
    solve(sp_nondist,solver=lsolver)
    @test abs(optimal_value(sp_nondist) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

@testset "$lsname Solver with Distributed Data: $name" for (lsolver,lsname) in lsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=lsolver)
    @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
end

info("Loading infeasible...")
include("infeasible.jl")
lsolvers = [(LShapedSolver(:dls,GurobiSolver(OutputFlag=0),log=false),"L-Shaped"),
            (LShapedSolver(:dtr,GurobiSolver(OutputFlag=0),crash=Crash.EVP(),autotune=true,log=false),"TR L-Shaped")]
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
