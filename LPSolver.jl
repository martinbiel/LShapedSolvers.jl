# LP problem, with assumed form
# min c^Tx
# s.t Ax = b
#     x >= 0
type LPProblem
    c::AbstractVector
    b::AbstractVector
    A::SparseMatrixCSC

    numRows::Int
    numCols::Int

    I::AbstractVector
    J::AbstractVector
    V::AbstractVector

    posVars
    negVars
    freeVars
    numSlacks::Int

    function LPProblem(m::JuMPModel)
        @assert JuMP.ProblemTraits(m).lin "Can only load linear models"
        prob = new()
        prob.numSlacks = 0
        loadStandardForm!(prob,m)
        return prob
    end
end

function loadStandardForm!(p::LPProblem,m::JuMPModel)
    p.numRows = length(m.linconstr)

    # Parse variables
    parseVariables!(p,m)

    # Build objective
    # ==============================
    addObjective!(p,m)

    # Build constraints
    # ==============================
    # Bound vector
    p.b = zeros(0)
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

function parseVariables!(p::LPProblem,m::JuMPModel)
    p.numCols = 0

    p.posVars = Dict{Any,Int}()
    p.negVars = Dict{Any,Int}()
    p.freeVars = Dict{Any,Tuple{Int,Int}}()

    j = 1
    for i = 1:m.numCols
        if m.colLower[i] == -Inf && m.colUpper[i] == Inf
            # Free variable
            p.freeVars[Variable(m,i)] = (j,j+1)
            j += 2
            p.numCols += 2
        elseif m.colLower[i] == -Inf && m.colUpper[i] == 0.0
            # Negative variable
            p.negVars[Variable(m,i)] = j
            j += 1
            p.numCols += 1
        elseif m.colLower[i] == 0.0 && m.colUpper[i] == Inf
            # Positive variable
            p.posVars[Variable(m,i)] = j
            j += 1
            p.numCols += 1
        else
            error("Can only handle simple variable bounds")
        end
    end
end

function addObjective!(p::LPProblem,m::JuMPModel)
    p.c = zeros(p.numCols)
    sign = m.objSense == :Min ? 1 : -1
    affobj = m.obj.aff
    @inbounds for (i,var) in enumerate(affobj.vars)
        @assert var.m == m "Variable not owned by model present in objective"
        if haskey(p.posVars,var)
            p.c[p.posVars[var]] += sign*affobj.coeffs[i]
        elseif haskey(p.negVars,var)
            p.c[p.negVars[var]] -= sign*affobj.coeffs[i]
        elseif haskey(p.freeVars,var)
            p.c[p.freeVars[var][1]] += sign*affobj.coeffs[i]
            p.c[p.freeVars[var][2]] -= sign*affobj.coeffs[i]
        else
            error("Variable $var has not been parsed")
        end
    end
end

function addRow!(p::LPProblem,m::JuMPModel,i::Int)
    constr = m.linconstr[i]
    coeffs = constr.terms.coeffs
    vars = constr.terms.vars

    @inbounds for (j,var) = enumerate(vars)
        if var.m == m
            push!(p.I,i)
            if haskey(p.posVars,var)
                push!(p.J,p.posVars[var])
                push!(p.V,coeffs[j])
            elseif haskey(p.negVars,var)
                push!(p.J,p.negVars[j])
                push!(p.V,-coeffs[j])
            elseif haskey(p.freeVars,var)
                push!(p.J,p.freeVars[var][1])
                push!(p.V,coeffs[j])
                push!(p.I,i)
                push!(p.J,p.freeVars[var][2])
                push!(p.V,-coeffs[j])
            else
                error("Variable $var has not been parsed")
            end
        end
    end

    if constr.lb != constr.ub
        p.numSlacks += 1
        if constr.lb == -Inf
            # upper bound
            push!(p.I,i)
            push!(p.J,p.numCols+p.numSlacks)
            push!(p.V,1)
            push!(p.b,constr.ub)
        elseif constr.ub == Inf
            # lower bound
            push!(p.I,i)
            push!(p.J,p.numCols+p.numSlacks)
            push!(p.V,-1)
            push!(p.b,constr.lb)
        else
            error("Can only standardize one sided constraints")
        end
        push!(p.c,0)
    else
        push!(p.b,constr.ub)
    end
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

type LPSolver
    lp::LPProblem

    obj::Real         # Objective
    x::AbstractVector # Primal variables
    λ::AbstractVector # Dual variables

    status::Symbol

    function LPSolver(lp::LPProblem)
        solver = new(lp)

        solver.obj = -Inf
        solver.x = similar(lp.c)
        solver.λ = similar(lp.b)

        return solver
    end
end

function (solver::LPSolver)()
    lp = solver.lp
    basis = collect((lp.numCols+1):(lp.numCols+lp.numSlacks))

    try
        x,obj,λ,s,_,_,_,status = primalsimplex(lp.A,lp.b,lp.c)

        solver.x = x
        solver.obj = obj
        solver.λ = λ
        solver.status = status
        updateSolution(solver)
    catch
        solver.status = :NotSolved
    end

    return nothing
end

status(solver::LPSolver) = solver.status

function updateSolution(solver::LPSolver)
    @assert status(solver) == :Optimal "Should not update unoptimized result"
    lp = solver.lp

    for (var,idx) in lp.posVars
        setvalue(var,solver.x[idx])
    end

    for (var,idx) in lp.negVars
        setvalue(var,-solver.x[idx])
    end

    for (var,idxs) in lp.freeVars
        i = idxs[1]
        j = idxs[2]
        setvalue(var,solver.x[i]-solver.x[j])
    end
end
