# LP problem, with assumed form
# min c^Tx
# s.t lb <= Ax <= ub
#     l <= x <= u
mutable struct LPProblem
    c::AbstractVector
    A::SparseMatrixCSC
    l::AbstractVector
    u::AbstractVector

    numRows::Int
    numCols::Int

    I::AbstractVector
    J::AbstractVector
    V::AbstractVector

    numSlacks::Int

    function LPProblem(m::JuMPModel)
        @assert JuMP.ProblemTraits(m).lin "Can only load linear models"
        prob = new()
        prob.numSlacks = 0
        loadStandardForm!(prob,m)
        return prob
    end

    function LPProblem(A::SparseMatrixCSC,
                       l::AbstractVector,
                       u::AbstractVector,
                       c::AbstractVector,
                       lb::AbstractVector,
                       ub::AbstractVector,
                       sense::Symbol)
        prob = new()
        prob.numSlacks = 0
        loadStandardForm!(prob,m)
        return prob
    end
end

function loadStandardForm!(p::LPProblem,m::JuMPModel)
    p.numRows = length(m.linconstr)
    p.numCols = m.numCols

    p.l = copy(m.colLower)
    p.u = copy(m.colUpper)

    # Build objective
    # ==============================
    addObjective!(p,m)

    # Build constraints
    # ==============================
    # Non-zero row indices
    p.I = Vector{Int}()
    # Non-zero column indices
    p.J = Vector{Int}()
    # Non-zero values
    p.V = Vector{Float64}()

    for i in 1:p.numRows
        addRow!(p,m,i)
    end

    p.A = sparse(p.I,p.J,p.V,p.numRows,p.numCols+p.numSlacks)
end

function loadStandardForm!(p::LPProblem,
                           A::SparseMatrixCSC,
                           l::AbstractVector,
                           u::AbstractVector,
                           c::AbstractVector,
                           lb::AbstractVector,
                           ub::AbstractVector,
                           sense::Symbol)
    p.numRows = size(A,1)
    p.numCols = size(A,2)

    p.l = copy(l)
    p.u = copy(u)

    # Build objective
    # ==============================
    p.c = sense == :Min ? copy(c) : -copy(c)

    # Build constraints
    # ==============================
    # Non-zero row indices
    p.I = Vector{Int}()
    # Non-zero column indices
    p.J = Vector{Int}()
    # Non-zero values
    p.V = Vector{Float64}()

    p.A = [A I]

    p.I,p,J,p.V = findnz(p.A)

    for i = 1:length(lb)
        p.numSlacks += 1
        push!(p.l,-ub[i])
        push!(p.u,-lb[i])
        push!(p.I,i)
        push!(p.J,p.numCols+p.numSlacks)
        push!(p.V,1)
        push!(p.c,0)
    end
end

function addObjective!(p::LPProblem,m::JuMPModel)
    p.c = zeros(p.numCols)
    sign = m.objSense == :Min ? 1 : -1
    affobj = m.obj.aff
    @inbounds for (i,var) in enumerate(affobj.vars)
        addToObjective!(p,m,var,sign*affobj.coeffs[i])
    end
end

function addToObjective!(p::LPProblem,m::JuMPModel,var::JuMPVariable,coeff::Real)
    p.c[var.col] += coeff
end

function addRow!(p::LPProblem,m::JuMPModel,i::Int)
    constr = m.linconstr[i]
    coeffs = constr.terms.coeffs
    vars = constr.terms.vars

    @inbounds for (j,var) = enumerate(vars)
        if var.m == m
            push!(p.I,i)
            push!(p.J,var.col)
            push!(p.V,coeffs[j])
        end
    end

    p.numSlacks += 1
    push!(p.l,-constr.ub)
    push!(p.u,-constr.lb)
    push!(p.I,i)
    push!(p.J,p.numCols+p.numSlacks)
    push!(p.V,1)
    push!(p.c,0)
end

function addRow!(p::LPProblem,a::AbstractVector,lb::Real,ub::Real,i::Int)

    for (j,coeff) in enumerate(a)
        push!(p.I,i)
        push!(p.J,j)
        push!(p.V,coeff)
    end

    # Slack
    p.numSlacks += 1
    push!(p.l,-ub)
    push!(p.u,-lb)
    push!(p.I,i)
    push!(p.J,p.numCols+p.numSlacks)
    push!(p.V,1)
    push!(p.c,0)
end

function addCols!(p::LPProblem,m::JuMPModel)
    newNumCols = m.numCols
    if (p.numCols-p.numSlacks) == newNumCols
        info("No new cols to add")
        return
    end

    # At this point it is necessary to reload the full model, to get indices right
    loadStandardForm!(p,m)
end

function addRows!(p::LPProblem,m::JuMPModel)
    newNumRows = length(m.linconstr)
    if p.numRows == newNumRows
        info("No new rows to add")
        return
    end

    for i = (p.numRows+1):newNumRows
        addRow!(p,m,i)
    end

    p.numRows = newNumRows
    p.A = sparse(p.I,p.J,p.V,p.numRows,p.numCols+p.numSlacks)
end

mutable struct LPSolver
    lp::LPProblem

    obj::Real         # Objective
    x::AbstractVector # Primal variables
    λ::AbstractVector # Constraint duals
    v::AbstractVector # Variable duals (lower bound)
    w::AbstractVector # Variable duals (upper bound)

    status::Symbol

    function LPSolver(lp::LPProblem)
        solver = new(lp)

        solver.obj = -Inf
        solver.x = similar(lp.c)
        solver.λ = zeros(size(lp.A,1))
        solver.v = zero(lp.c)
        solver.w = zero(lp.c)

        solver.status = :NotSolved

        return solver
    end
end

function (solver::LPSolver)()
    lp = solver.lp
    basis = collect((lp.numCols+1):(lp.numCols+lp.numSlacks))

    env = Gurobi.Env()
    #x,obj,λ,s,_,_,_,status = primalsimplex(lp.A,lp.b,lp.c)
    sol = linprog(lp.c, lp.A, fill('=',lp.numRows), 0.0, lp.l, lp.u, GurobiSolver(env, OutputFlag=0))

    solver.status = sol.status

    if solver.status == :Optimal
        solver.x = sol.sol
        solver.obj = sol.objval
        solver.λ = sol.attrs[:lambda]
        solver.v,solver.w = getVariableDuals(solver,sol.attrs[:redcost])
    elseif solver.status == :Infeasible
        solver.obj = Inf
        solver.λ = sol.attrs[:infeasibilityray]
    elseif solver.status == :Unbounded
        solver.obj = -Inf
    else
        error(string("LP could not be solved, returned status: ",solver.status))
    end

    return nothing
end

function getVariableDuals(solver::LPSolver,d::AbstractVector)
    l = solver.lp.l
    v = zero(l)
    u = solver.lp.u
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

status(solver::LPSolver) = solver.status

function updateSolution(solver::LPSolver,model::JuMPModel)
    @assert status(solver) == :Optimal "Should not update unoptimized result"
    numCols = solver.lp.numCols
    model.colVal = solver.x[1:numCols]
end
