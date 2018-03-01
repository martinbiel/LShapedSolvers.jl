using LShapedSolvers
using Base.Test
using JuMP
using StructJuMP
using Clp
using Gurobi

function simplemodel()
    p = [0.4,0.6]
    d = [500 100; 300 300]
    q = [-24 -28; -28 -32]

    numScen = 2
    m = StructuredModel(num_scenarios=numScen)
    @variable(m, x1 >= 40)
    @variable(m, x2 >= 20)
    @objective(m, Min, 100*x1 + 150*x2)
    @constraint(m, x1+x2 <= 120)

    for i in 1:numScen
        scen = StructuredModel(parent=m, id=i, prob=p[i])
        @variable(scen, 0 <= y1 <= d[i,1])
        @variable(scen, 0 <= y2 <= d[i,2])
        @objective(scen, Min, q[i,1]*y1 + q[i,2]*y2)
        @constraint(scen, 6*y1 + 10*y2 <= 60*x1)
        @constraint(scen, 8*y1 + 5*y2 <= 80*x2)
    end

    ref = Model(solver=ClpSolver())
    @variable(ref, x[1:2])
    @variable(ref, y[1:2,1:2])
    @objective(ref, Min, 100*x[1] + 150*x[2] + p[1]*(q[1,1]*y[1,1]+q[1,2]*y[1,2]) + p[2]*(q[2,1]*y[2,1] + q[2,2]*y[2,2]))
    @constraint(ref, x[1]+x[2] <= 120)
    @constraint(ref, x[1] >= 40)
    @constraint(ref, x[2] >= 20)

    for i in 1:2
        @constraint(ref, 6*y[i,1] + 10*y[i,2] <= 60*x[1])
        @constraint(ref, 8*y[i,1] + 5*y[i,2] <= 80*x[2])
        @constraint(ref, y[i,1] <= d[i,1])
        @constraint(ref, y[i,2] <= d[i,2])
    end

    return m, ref
end

@testset "L-Shaped Solver" begin
    m,ref = simplemodel()
    solver = ClpSolver()
    solve(ref)

    ls = LShaped(m,solver,solver)
    ls()
    @test norm(LShapedSolvers.get_solution(ls) - ref.colVal[1:2]) <= 1e-6
    @test abs(get_objective_value(ls) - ref.objVal) <= 1e-6
end

@testset "L-Shaped Solver with Regularization" begin
    m,ref = simplemodel()
    solver = Gurobi.GurobiSolver(OutputFlag=0)
    solve(ref)

    ls = Regularized(m,solver,solver)
    ls()
    @test norm(LShapedSolvers.get_solution(ls) - ref.colVal[1:2]) <= 1e-6
    @test abs(get_objective_value(ls) - ref.objVal) <= 1e-6
end

@testset "L-Shaped Solver with Trust-Region" begin
    m,ref = simplemodel()
    solver = ClpSolver()
    solve(ref)

    ls = TrustRegion(m,[40,20.],solver,solver)
    ls()
    @test norm(LShapedSolvers.get_solution(ls) - ref.colVal[1:2]) <= 1e-6
    @test abs(get_objective_value(ls) - ref.objVal) <= 1e-6
end
