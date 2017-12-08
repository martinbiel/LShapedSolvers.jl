struct SubProblem{float_t <: Real, array_t <: AbstractVector, solver_t <: LQSolver}
    id::Int
    π::float_t

    solver::solver_t

    h::Tuple{array_t,array_t}
    x::array_t
    y::array_t
    masterterms::Vector{Tuple{Int,Int,float_t}}

    function (::Type{SubProblem})(model::JuMPModel,parent::JuMPModel,id::Integer,π::Real,x::AbstractVector,y₀::AbstractVector,optimsolver::AbstractMathProgSolver)
        float_t = promote_type(eltype(x),eltype(y₀),Float32)
        x_ = convert(AbstractVector{float_t},x)
        y₀_ = convert(AbstractVector{float_t},y₀)
        array_t = typeof(x)

        solver = LQSolver(model,optimsolver)

        subproblem = new{float_t,array_t,typeof(solver)}(id,
                                                         π,
                                                         solver,
                                                         (convert(array_t,getconstrLB(solver.lqmodel)),
                                                          convert(array_t,getconstrUB(solver.lqmodel))),
                                                         x_,
                                                         y₀_,
                                                         Vector{Tuple{Int,Int,float_t}}()
                                                         )
        parseSubProblem!(subproblem,model,parent)

        return subproblem
    end
end

function parseSubProblem!(subproblem::SubProblem,model::JuMPModel,parent::JuMPModel)
    for (i,constr) in enumerate(model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == parent
                # var is a first stage variable
                push!(subproblem.masterterms,(i,var.col,-constr.terms.coeffs[j]))
            end
        end
    end
end

function update_subproblem!(subproblem::SubProblem,x::AbstractVector)
    lb = getconstrLB(subproblem.solver.lqmodel)
    ub = getconstrUB(subproblem.solver.lqmodel)
    for i in [term[1] for term in unique(term -> term[1],subproblem.masterterms)]
        lb[i] = subproblem.h[1][i]
        ub[i] = subproblem.h[2][i]
    end
    for (i,j,coeff) in subproblem.masterterms
        lb[i] += coeff*x[j]
        ub[i] += coeff*x[j]
    end
    setconstrLB!(subproblem.solver.lqmodel, lb)
    setconstrUB!(subproblem.solver.lqmodel, ub)
    subproblem.x[:] = x
end
update_subproblems!(subproblems::Vector{<:SubProblem},x::AbstractVector) = map(prob -> update_subproblem!(prob,x),subproblems)

function (subproblem::SubProblem)()
    subproblem.solver(subproblem.y)
    solvestatus = status(subproblem.solver)
    if solvestatus == :Optimal
        subproblem.y[:] = getsolution(subproblem.solver)
        return OptimalityCut(subproblem)
    elseif solvestatus == :Infeasible
        return FeasibilityCut(subproblem)
    elseif lshaped.status == :Unbounded
        return Unbounded(subproblem)
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subproblem.id,string(solvestatus)))
    end
end

function (subproblem::SubProblem)(x::AbstractVector)
    updateSubProblem!(subproblem,x)
    subproblem.solver(subproblem.y)
    solvestatus = status(subproblem.solver)
    if solvestatus == :Optimal
        y[:] = getsolution(subproblem.solver)
        return getobjval(subproblem.solver)
    elseif solvestatus == :Infeasible
        error(@sprintf("Subproblem %d is infeasible at the given first-stage variable",subproblem.id))
    elseif lshaped.status == :Unbounded
        error(@sprintf("Subproblem %d is unbounded at the given first-stage variable",subproblem.id))
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subproblem.id,string(solvestatus)))
    end
end
