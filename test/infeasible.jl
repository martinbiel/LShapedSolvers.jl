@everywhere begin
    struct InfeasibleScenario <: AbstractScenarioData
        π::Probability
        ξ::Vector{Float64}
    end

    function StochasticPrograms.expected(scenarios::Vector{InfeasibleScenario})
        isempty(scenarios) && return InfeasibleScenario(1.,zeros(2))
        sd = InfeasibleScenario(1.,sum([s.π*s.ξ for s in scenarios]))
    end
end

s1 = InfeasibleScenario(0.5,[6,8])
s2 = InfeasibleScenario(0.5,[4,4])

sds = [s1,s2]

infeasible = StochasticProgram(sds)

@first_stage infeasible = begin
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)
    @objective(model, Min, 3*x1 + 2*x2)
end

@second_stage infeasible = begin
    @decision x1 x2
    s = scenario
    @variable(model, 0.8*s.ξ[1] <= y1 <= s.ξ[1])
    @variable(model, 0.8*s.ξ[2] <= y2 <= s.ξ[2])
    @objective(model, Min, -15*y1 - 12*y2)
    @constraint(model, 3*y1 + 2*y2 <= x1)
    @constraint(model, 2*y1 + 5*y2 <= x2)
end
