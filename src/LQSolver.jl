mutable struct LQSolver{M <: MPB.AbstractLinearQuadraticModel, S <: MPB.AbstractMathProgSolver}
    lqmodel::M
    optimsolver::S

    function (::Type{LQSolver})(model::JuMP.Model,optimsolver::MPB.AbstractMathProgSolver)
        lqmodel = MPB.LinearQuadraticModel(optimsolver)
        solver = new{typeof(lqmodel),typeof(optimsolver)}(lqmodel,optimsolver)
        MPB.loadproblem!(solver.lqmodel,loadLP(model)...)
        return solver
    end

    function (::Type{LQSolver})(model::JuMP.Model,optimsolver::MPB.AbstractMathProgSolver, ::Val{false})
        lqmodel = MPB.LinearQuadraticModel(optimsolver)
        return new{typeof(lqmodel),typeof(optimsolver)}(lqmodel,optimsolver)
    end

    function (::Type{LQSolver})(model::JuMP.Model,optimsolver::MPB.AbstractMathProgSolver, ::Val{true})
        lqmodel = MPB.LinearQuadraticModel(optimsolver)
        solver = new{typeof(lqmodel),typeof(optimsolver)}(lqmodel,optimsolver)
        MPB.loadproblem!(solver.lqmodel,loadLP(model)...)
        feasibility_problem!(solver)
        return solver
    end
end

function (solver::LQSolver)(x₀::AbstractVector)
    if applicable(MPB.setwarmstart!,solver.lqmodel,x₀)
        try
            MPB.setwarmstart!(solver.lqmodel,x₀)
        catch
            @warn "Could not set warm start for some reason"
        end
    end
    MPB.optimize!(solver.lqmodel)

    optimstatus = MPB.status(solver.lqmodel)
    !(optimstatus ∈ [:Optimal,:Infeasible,:Unbounded]) && error("LP could not be solved, returned status: ", optimstatus)

    return nothing
end

function getsolution(solver::LQSolver)
    n = MPB.numvar(solver.lqmodel)
    optimstatus = MPB.status(solver.lqmodel)
    if optimstatus == :Optimal
        return MPB.getsolution(solver.lqmodel)
    else
        @warn "LP was not solved to optimality, returned status: $optimstatus"
        return fill(NaN, n)
    end
end

function getobjval(solver::LQSolver)
    optimstatus = MPB.status(solver.lqmodel)
    if optimstatus == :Optimal
        return MPB.getobjval(solver.lqmodel)
    elseif optimstatus == :Infeasible
        return Inf
    elseif optimstatus == :Unbounded
        return -Inf
    else
        @warn "LP could not be solved, returned status: $optimstatus"
        return NaN
    end
end

function getredcosts(solver::LQSolver)
    cols = MPB.numvar(solver.lqmodel)
    optimstatus = MPB.status(solver.lqmodel)
    if optimstatus == :Optimal
        return try
            MPB.getreducedcosts(solver.lqmodel)[1:cols]
        catch
            fill(NaN, cols)
        end
    else
        @warn "LP was not solved to optimality. Return status: $optimstatus"
        return fill(NaN, cols)
    end
end

function getduals(solver::LQSolver)
    rows = MPB.numconstr(solver.lqmodel)
    optimstatus = MPB.status(solver.lqmodel)
    if optimstatus == :Optimal
        return try
            MPB.getconstrduals(solver.lqmodel)[1:rows]
        catch
            fill(NaN, rows)
        end
    elseif optimstatus ==:Infeasible
        return try
            MPB.getinfeasibilityray(solver.lqmodel)[1:rows]
        catch
            fill(NaN, rows)
        end
    else
        @warn "LP was not solved to optimality, and the model was not infeasible. Return status: $optimstatus"
        return fill(NaN, rows)
    end
end

status(solver::LQSolver) = MPB.status(solver.lqmodel)

function feasibility_problem!(solver::LQSolver)
    MPB.setobj!(solver.lqmodel,zeros(MPB.numvar(solver.lqmodel)))
    for i = 1:MPB.numconstr(solver.lqmodel)
        MPB.addvar!(solver.lqmodel,[i],[1.0],0.0,Inf,1.0)
        MPB.addvar!(solver.lqmodel,[i],[-1.0],0.0,Inf,1.0)
    end
end

function loadLP(m::JuMP.Model)
    l = copy(m.colLower)
    u = copy(m.colUpper)

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
    l = MPB.getvarLB(solver.lqmodel)
    v = zero(l)
    u = MPB.getvarUB(solver.lqmodel)
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
    λ = MPB.getconstrduals(solver.lqmodel)
    Ax = MPB.getconstrsolution(solver.lqmodel)

    lb = MPB.getconstrLB(solver.lqmodel)
    λl = zero(lb)
    ub = MPB.getconstrUB(solver.lqmodel)
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
