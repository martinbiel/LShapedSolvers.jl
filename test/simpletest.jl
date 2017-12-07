using StructJuMP
using Clp
using Base.Test
push!(LOAD_PATH,"/home/mbiel/projects/LShaped/")
using LShaped

function simplemodel()
    p = [0.4,0.6]
    d = [500 100; 300 300]
    q = [-24 -28; -28 -32]

    numScen = 2
    m = StructuredModel(num_scenarios=numScen)
    @variable(m, x[1:2])
    @objective(m, Min, 100*x[1] + 150*x[2])
    @constraint(m, x[1]+x[2] <= 120)
    @constraint(m, x[1] >= 40)
    @constraint(m, x[2] >= 20)

    for i in 1:numScen
        scen = StructuredModel(parent=m, id=i, prob=p[i])
        @variable(scen, y[1:2] >= 0)
        @objective(scen, Min, q[i,1]*y[1] + q[i,2]*y[2])
        @constraint(scen, 6*y[1] + 10*y[2] <= 60*x[1])
        @constraint(scen, 8*y[1] + 5*y[2] <= 80*x[2])
        @constraint(scen, y[1] <= d[i,1])
        @constraint(scen, y[2] <= d[i,2])
    end
    return m
end

info("Starting simple test...")

@testset "Simple test" begin

    m = simplemodel()
    solver = LShapedSolver(m)

end

info("Simple test finished")
