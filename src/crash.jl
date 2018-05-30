module Crash

using StochasticPrograms
using MathProgBase

abstract type CrashMethod end

struct None <: CrashMethod end

function (crash::None)(sp::JuMP.Model,solver::MathProgBase.AbstractMathProgSolver)
    return rand(sp.numCols)
end

struct EVP <: CrashMethod end

function (crash::EVP)(sp::JuMP.Model,solver::MathProgBase.AbstractMathProgSolver)
    evp = StochasticPrograms.EVP(sp,solver)
    status = solve(evp)
    status != :Optimal && error("Could not solve EVP model during crash procedure. Aborting.")
    return evp.colVal[1:sp.numCols]
end

struct Scenario{S <: AbstractScenarioData} <: CrashMethod
    scenario::S

    function (::Type{Scenario})(scenario::S) where S <: AbstractScenarioData
        return new{S}(scenario)
    end
end

function (crash::Scenario)(sp::JuMP.Model,solver::MathProgBase.AbstractMathProgSolver)
    ws = WS(sp,crash.scenario,solver)
    status = solve(ws)
    status != :Optimal && error("Could not solve wait-and-see model during crash procedure. Aborting.")
    return ws.colVal[1:sp.numCols]
end

struct Custom{T <: Real} <: CrashMethod
    x₀::Vector{T}

    function (::Type{Custom})(x₀::Vector{T}) where T <: Real
        return new{T}(x₀)
    end
end

function (crash::Custom)(sp::JuMP.Model,solver::MathProgBase.AbstractMathProgSolver)
    return x₀
end

end
