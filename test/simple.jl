@scenario Simple = begin
    q₁::Float64
    q₂::Float64
    d₁::Float64
    d₂::Float64
end

s₁ = SimpleScenario(-24.0, -28.0, 500.0, 100.0, probability = 0.4)
s₂ = SimpleScenario(-28.0, -32.0, 300.0, 300.0, probability = 0.6)

simple = StochasticProgram([s₁,s₂])

@first_stage simple = begin
    @variable(model, x₁ >= 40)
    @variable(model, x₂ >= 20)
    @objective(model, Min, 100*x₁ + 150*x₂)
    @constraint(model, x₁ + x₂ <= 120)
end

@second_stage simple = begin
    @decision x₁ x₂
    ξ = scenario
    @variable(model, 0 <= y₁ <= ξ.d₁)
    @variable(model, 0 <= y₂ <= ξ.d₂)
    @objective(model, Min, ξ.q₁*y₁ + ξ.q₂*y₂)
    @constraint(model, 6*y₁ + 10*y₂ <= 60*x₁)
    @constraint(model, 8*y₁ + 5*y₂ <= 80*x₂)
end

push!(problems,(simple,"Simple"))
