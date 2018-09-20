mutable struct LQSolver{M <: AbstractLinearQuadraticModel, S <: AbstractMathProgSolver}
    lqmodel::M
    optimsolver::S

    function (::Type{LQSolver})(model::JuMP.Model,optimsolver::AbstractMathProgSolver)
        lqmodel = LinearQuadraticModel(optimsolver)
        solver = new{typeof(lqmodel),typeof(optimsolver)}(lqmodel,optimsolver)
        loadproblem!(solver.lqmodel,loadLP(model)...)

        return solver
    end
end

function (solver::LQSolver)(x₀::AbstractVector)
    if applicable(setwarmstart!,solver.lqmodel,x₀)
        try
            setwarmstart!(solver.lqmodel,x₀)
        catch
            warn("Could not set warm start for some reason")
        end
    end
    optimize!(solver.lqmodel)

    optimstatus = status(solver)
    !(status(solver) ∈ [:Optimal,:Infeasible,:Unbounded]) && error("LP could not be solved, returned status: ", optimstatus)

    return nothing
end

function getsolution(solver::LQSolver)
    n = numvar(solver.lqmodel)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return getsolution(solver.lqmodel)
    else
        warn("LP was not solved to optimality, returned status: ", optimstatus)
        return fill(NaN, n)
    end
end

function getobjval(solver::LQSolver)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return getobjval(solver.lqmodel)
    elseif optimstatus == :Infeasible
        return Inf
    elseif optimstatus == :Unbounded
        return -Inf
    else
        warn("LP could not be solved, returned status: ", optimstatus)
        return NaN
    end
end

function getredcosts(solver::LQSolver)
    cols = numvar(solver.lqmodel)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return try
            getreducedcosts(solver.lqmodel)[1:cols]
        catch
            fill(NaN, cols)
        end
    else
        warn("LP was not solved to optimality. Return status: ", optimstatus)
        return fill(NaN, cols)
    end
end

function getduals(solver::LQSolver)
    rows = numconstr(solver.lqmodel)
    optimstatus = status(solver)
    if optimstatus == :Optimal
        return try
            getconstrduals(solver.lqmodel)[1:rows]
        catch
            fill(NaN, rows)
        end
    elseif optimstatus ==:Infeasible
        return try
            getinfeasibilityray(solver.lqmodel)[1:rows]
        catch
            fill(NaN, rows)
        end
    else
        warn("LP was not solved to optimality, and the model was not infeasible. Return status: ", optimstatus)
        return fill(NaN, rows)
    end
end

status(solver::LQSolver) = MathProgBase.SolverInterface.status(solver.lqmodel)

function feasibility_problem!(solver::LQSolver)
    setobj!(solver.lqmodel,zeros(numvar(solver.lqmodel)))
    for i = 1:numconstr(solver.lqmodel)
        addvar!(solver.lqmodel,[i],[1.0],0.0,Inf,1.0)
        addvar!(solver.lqmodel,[i],[-1.0],0.0,Inf,1.0)
    end
end

function loadLP(m::JuMP.Model)
    l = m.colLower
    u = m.colUpper

    # Build objective
    # ==============================
    c = JuMP.prepAffObjective(m)
    c *= m.objSense == :Min ? 1 : -1

    # Build constraints
    # ==============================
    # Non-zero row indices
    I = Vector{Int}()
    # Non-zero column indices
    J = Vector{Int}()
    # Non-zero values
    V = Vector{Float64}()
    # Lower constraint bound
    lb = zeros(Float64,length(m.linconstr))
    # Upper constraint bound
    ub = zeros(Float64,length(m.linconstr))

    for (i,constr) in enumerate(m.linconstr)
        coeffs = constr.terms.coeffs
        vars = constr.terms.vars
        @inbounds for (j,var) = enumerate(vars)
            if var.m == m
                push!(I,i)
                push!(J,var.col)
                push!(V,coeffs[j])
            end
        end
        lb[i] = constr.lb
        ub[i] = constr.ub
    end
    A = sparse(I,J,V,length(m.linconstr),m.numCols)

    return A,l,u,c,lb,ub,:Min
end

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
