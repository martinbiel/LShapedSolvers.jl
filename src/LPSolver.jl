struct LQSolver{float_t <: Real,array_t <: AbstractVector,model_t <: AbstractLinearQuadraticModel, solver_t <: AbstractMathProgSolver,}
    x::array_t          # Primal variables
    λ::array_t          # Constraint duals
    v::array_t          # Variable duals (lower bound)
    w::array_t          # Variable duals (upper bound)

    lqmodel::model_t
    optimsolver::solver_t

    function (::Type{LQSolver})(model::JuMPModel,optimsolver::AbstractMathProgSolver,x₀::AbstractVector)
        if length(x₀) != model.numCols
            throw(ArgumentError(string("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)))
        end

        float_t = promote_type(eltype(x₀),Float32)
        x₀_ = convert(AbstractVector{float_t},x₀)
        array_t = typeof(x₀_)

        lqmodel = LinearQuadraticModel(optimsolver)
        solver = new{float_t,array_t,typeof(lqmodel),typeof(optimsolver)}(
            x₀_,
            convert(array_t,zeros(length(model.linconstr))),
            convert(array_t,zeros(model.numCols)),
            convert(array_t,zeros(model.numCols)),
            lqmodel,
            optimsolver,
        )
        loadproblem!(solver.lqmodel,loadLP(model)...)

        return solver
    end
end
LQSolver(model::JuMPModel,optimsolver::AbstractMathProgSolver) = LQSolver(model,optimsolver,rand(model.numCols))

function (solver::LQSolver)()
    if applicable(setwarmstart!,solver.lqmodel,solver.x)
        setwarmstart!(solver.lqmodel,solver.x)
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
