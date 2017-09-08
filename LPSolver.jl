abstract type AbstractLQSolver end

status(solver::AbstractLQSolver) = solver.status

function getVariableDuals(solver::AbstractLQSolver,d::AbstractVector)
    l = getvarLB(solver.model)
    v = zero(l)
    u = getvarUB(solver.model)
    w = zero(u)
    for i = 1:length(l)
        if l[i] > -Inf
            if u[i] == Inf
                v[i] = d[i]
            else
                if solver.x[i] == l[i]
                    v[i] = d[i]
                elseif solver.x[i] == u[i]
                    w[i] = d[i]
                else
                    continue
                end
            end
        else
            if u[i] < Inf
                w[i] = d[i]
            end
        end
    end
    return v,w
end

function getConstrDuals(solver::AbstractLQSolver)
    λ = getconstrduals(solver.model)
    Ax = getconstrsolution(solver.model)
    lb = getconstrLB(solver.model)
    λl = zero(lb)
    ub = getconstrUB(solver.model)
    λu = zero(ub)
    for i = 1:length(lb)
        if lb[i] > -Inf
            if ub[i] == Inf
                λl[i] = λ[i]
            else
                if Ax[i] == lb[i]
                    λl[i] = λ[i]
                elseif Ax[i] == ub[i]
                    λu = λ[i]
                else
                    continue
                end
            end
        else
            if ub[i] < Inf
                λu[i] = λ[i]
            end
        end
    end
    return λl,λu
end

mutable struct LQSolver <: AbstractLQSolver
    obj::Real         # Objective
    x::AbstractVector # Primal variables
    λ::AbstractVector # Constraint duals
    v::AbstractVector # Variable duals (lower bound)
    w::AbstractVector # Variable duals (upper bound)

    model::AbstractLinearQuadraticModel
    solver::AbstractMathProgSolver
    status::Symbol

    function LQSolver(model::JuMPModel,optimsolver=GurobiSolver(OutputFlag=0))
        solver = new()

        solver.obj = -Inf
        solver.x = zeros(model.numCols)
        solver.λ = zeros(length(model.linconstr))
        solver.v = zeros(model.numCols)
        solver.w = zeros(model.numCols)

        solver.status = :NotSolved
        solver.solver = optimsolver
        solver.model = LinearQuadraticModel(solver.solver)
        loadproblem!(solver.model,loadLP(model)...)

        return solver
    end
end

function (solver::LQSolver)()
    setwarmstart!(solver.model,solver.x)
    optimize!(solver.model)
    solver.status = status(solver.model)

    if solver.status == :Optimal
        solver.x = getsolution(solver.model)
        solver.obj = getobjval(solver.model)
        solver.λ = getconstrduals(solver.model)
        solver.v,solver.w = getVariableDuals(solver,getreducedcosts(solver.model))
    elseif solver.status == :Infeasible
        solver.obj = Inf
        solver.λ = getinfeasibilityray(solver.model)
    elseif solver.status == :Unbounded
        solver.obj = -Inf
    else
        error(string("LP could not be solved, returned status: ",solver.status))
    end

    return nothing
end
