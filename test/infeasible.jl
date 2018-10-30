@scenario Infeasible = begin
    ξ₁::Float64
    ξ₂::Float64
end

s₁ = InfeasibleScenario(6., 8., probability = 0.5)
s₂ = InfeasibleScenario(4.0, 4.0, probability = 0.5)

infeasible = StochasticProgram([s₁,s₂])

@first_stage infeasible = begin
    @variable(model, x₁ >= 0)
    @variable(model, x₂ >= 0)
    @objective(model, Min, 3*x₁ + 2*x₂)
end

@second_stage infeasible = begin
    @decision x₁ x₂
    s = scenario
    @variable(model, 0.8*s.ξ₁ <= y₁ <= s.ξ₁)
    @variable(model, 0.8*s.ξ₂ <= y₂ <= s.ξ₂)
    @objective(model, Min, -15*y₁ - 12*y₂)
    @constraint(model, 3*y₁ + 2*y₂ <= x₁)
    @constraint(model, 2*y₁ + 5*y₂ <= x₂)
end
