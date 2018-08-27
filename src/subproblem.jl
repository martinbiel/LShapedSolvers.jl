struct SubProblem{T <: Real, A <: AbstractVector, S <: LQSolver}
    id::Int
    π::T

    solver::S

    h::Tuple{A,A}
    x::A
    y::A
    masterterms::Vector{Tuple{Int,Int,T}}

    function (::Type{SubProblem})(model::JuMP.Model,id::Integer,π::Real,x::AbstractVector,y₀::AbstractVector,masterterms::Vector{Tuple{Int,Int,R}},optimsolver::AbstractMathProgSolver) where R <: Real
        T = promote_type(eltype(x),eltype(y₀),R,Float32)
        x_ = convert(AbstractVector{T},x)
        y₀_ = convert(AbstractVector{T},y₀)
        masterterms_ = convert(Vector{Tuple{Int,Int,T}},masterterms)
        A = typeof(x_)

        solver = LQSolver(model,optimsolver)

        subproblem = new{T,A,typeof(solver)}(id,
                                             π,
                                             solver,
                                             (convert(A,getconstrLB(solver.lqmodel)),
                                              convert(A,getconstrUB(solver.lqmodel))),
                                             x_,
                                             y₀_,
                                             masterterms_
                                             )

        return subproblem
    end

    function (::Type{SubProblem})(model::JuMP.Model,parent::JuMP.Model,id::Integer,π::Real,x::AbstractVector,y₀::AbstractVector,optimsolver::AbstractMathProgSolver)
        T = promote_type(eltype(x),eltype(y₀),Float32)
        x_ = convert(AbstractVector{T},x)
        y₀_ = convert(AbstractVector{T},y₀)
        A = typeof(x_)

        solver = LQSolver(model,optimsolver)

        subproblem = new{T,A,typeof(solver)}(id,
                                             π,
                                             solver,
                                             (convert(A,getconstrLB(solver.lqmodel)),
                                              convert(A,getconstrUB(solver.lqmodel))),
                                             x_,
                                             y₀_,
                                             Vector{Tuple{Int,Int,T}}()
                                             )
        parse_subproblem!(subproblem,model,parent)

        return subproblem
    end
end

function parse_subproblem!(subproblem::SubProblem,model::JuMP.Model,parent::JuMP.Model)
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

function get_solution(subproblem::SubProblem)
    return copy(subproblem.y),getredcosts(subproblem.solver),getduals(subproblem.solver),getobjval(subproblem.solver)
end

function (subproblem::SubProblem)()
    subproblem.solver(subproblem.y)
    solvestatus = status(subproblem.solver)
    if solvestatus == :Optimal
        subproblem.y[:] = getsolution(subproblem.solver)
        return OptimalityCut(subproblem)
    elseif solvestatus == :Infeasible
        return FeasibilityCut(subproblem)
    elseif solvestatus == :Unbounded
        return Unbounded(subproblem)
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subproblem.id,string(solvestatus)))
    end
end

function (subproblem::SubProblem)(x::AbstractVector)
    update_subproblem!(subproblem,x)
    subproblem.solver(subproblem.y)
    solvestatus = status(subproblem.solver)
    if solvestatus == :Optimal
        subproblem.y[:] = getsolution(subproblem.solver)
        return getobjval(subproblem.solver)
    elseif solvestatus == :Infeasible
        error(@sprintf("Subproblem %d is infeasible at the given first-stage variable",subproblem.id))
    elseif solvestatus == :Unbounded
        error(@sprintf("Subproblem %d is unbounded at the given first-stage variable",subproblem.id))
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subproblem.id,string(solvestatus)))
    end
end
