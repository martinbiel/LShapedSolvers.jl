type SubProblem
    model::JuMPModel
    problem::LPProblem
    solver::LPSolver
    parent # Parent LShaped

    updates # First stage variables infer updates on subproblem bounds

    function SubProblem(m::JuMPModel)
        subprob = new(m)

        p = LPProblem(m)
        subprob.problem = p
        subprob.solver = LPSolver(p)

        return subprob
    end
end

type LShapedSolver

    structuredModel::JuMPModel

    masterProblem::LPProblem
    masterSolver::LPSolver

    numScenarios::Int
    subProblems::Vector{SubProblem}

    # Cuts
    θ
    numCuts::Int

    function LShapedSolver(m::JuMPModel)
        @assert haskey(m.ext,:Stochastic) "The provided model is not structured"
        lshaped = new(m)

        prepareMaster!(lshaped)

        p = LPProblem(m)
        lshaped.masterProblem = p
        lshaped.masterSolver = LPSolver(p)

        n = num_scenarios(m)
        lshaped.numScenarios = n
        lshaped.subProblems = Vector{SubProblem}()
        for i = 1:n
            subprob = SubProblem(lshaped,getchildren(m)[i])
            push!(lshaped.subProblems,subprob)
        end

        lshaped.numCuts = 0

        return lshaped
    end
end

function prepareMaster!(lshaped::LShapedSolver)
    lshaped.θ = @variable(lshaped.structuredModel,θ)
end

function SubProblem(parent::LShapedSolver,m::JuMPModel)
    subprob = SubProblem(m)

    subprob.parent = parent

    subprob.updates = []
    parseSubProblem!(subprob)

    return subprob
end

function parseSubProblem!(subprob::SubProblem)
    for (i,constr) in enumerate(subprob.model.linconstr)
        for (j,var) in enumerate(constr.terms.vars)
            if var.m == subprob.parent.structuredModel
                # var is a first stage variable
                push!(subprob.updates,(i,var,-constr.terms.coeffs[j]))
            end
        end
    end
end

function updateSubProblem!(subprob::SubProblem)
    @assert status(subprob.parent.masterSolver) == :Optimal

    for (i,x,coeff) in subprob.updates
        constr = subprob.model.linconstr[i]
        rhs = coeff*getvalue(x)
        if constr.lb == -Inf
            rhs += constr.ub
        elseif constr.ub == Inf
            rhs += constr.lb
        else
            error("Can only handle one sided constraints")
        end
        subprob.problem.b[i] = ub + coeff*getvalue(x)
    end
end
