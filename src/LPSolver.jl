struct LQSolver{model_t <: AbstractLinearQuadraticModel, solver_t <: AbstractMathProgSolver}
    lqmodel::model_t
    optimsolver::solver_t

    function (::Type{LQSolver})(model::JuMPModel,optimsolver::AbstractMathProgSolver)
        lqmodel = LinearQuadraticModel(optimsolver)
        solver = new{typeof(lqmodel),typeof(optimsolver)}(lqmodel,optimsolver)
        loadproblem!(solver.lqmodel,loadLP(model)...)

        return solver
    end
end

function (solver::LQSolver)(x₀::AbstractVector)
    if applicable(setwarmstart!,solver.lqmodel,x₀)
        setwarmstart!(solver.lqmodel,x₀)
    end
    optimize!(solver.lqmodel)

    optimstatus = status(solver)
    !(status(solver) ∈ [:Optimal,:Infeasible,:Unbounded]) && error("LP could not be solved, returned status: ", optimstatus)

    return nothing
end

function getsolution(solver::LQSolver)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return getsolution(solver.lqmodel)
    else
        error("LP was not solved to optimality, returned status: ", optimstatus)
    end
end

function getobjval(solver::LQSolver)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return getobjval(solver.lqmodel)
    elseif optimstatus ==:Infeasible
        return Inf
    elseif status == :Unbounded
        solver.obj = -Inf
    else
        error("LP could not be solved, returned status: ", optimstatus)
    end
end

function getduals(solver::LQSolver)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return getconstrduals(solver.lqmodel)
    elseif optimstatus ==:Infeasible
        return getinfeasibilityray(solver.lqmodel)
    else
        error("LP was not solved to optimality, and the model was not infeasible. Returned status: ", optimstatus)
    end
end

status(solver::LQSolver) = MathProgBase.SolverInterface.status(solver.lqmodel)

function getVariableDuals(solver::LQSolver,d::AbstractVector)
    l = getvarLB(solver.lqmodel)
    v = zero(l)
    u = getvarUB(solver.lqmodel)
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

function getConstrDuals(solver::LQSolver)
    λ = getconstrduals(solver.lqmodel)
    Ax = getconstrsolution(solver.lqmodel)

    lb = getconstrLB(solver.lqmodel)
    λl = zero(lb)
    ub = getconstrUB(solver.lqmodel)
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
