mutable struct TrustRegionLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterSolver::AbstractLQSolver
    x::AbstractVector
    obj::Real

    committee::Vector{AbstractHyperplane}
    inactive::Vector{AbstractHyperplane}
    violating::PriorityQueue

    nscenarios::Integer
    subProblems::Vector{SubProblem}
    subObjectives::AbstractVector

    # Trust region
    ξ::AbstractVector
    Q̃::Real
    Q̃_hist::AbstractVector
    Δ::Real
    Δ_hist::AbstractVector
    Δ̅::Real
    cΔ::Integer
    nMajorSteps::Integer
    nMinorSteps::Integer

    # Cuts
    θs::AbstractVector
    nOptimalityCuts::Integer
    nFeasibilityCuts::Integer
    cuts::Vector{AbstractHyperplane}

    status::Symbol
    γ::Real
    τ::Real

    function TrustRegionLShapedSolver(m::JuMPModel,ξ::Vector{Float64})
        lshaped = new(m)

        if length(ξ) != m.numCols
            throw(ArgumentError(string("Incorrect length of regularizer, has ",length(ξ)," should be ",m.numCols)))
        end

        lshaped.x = ξ
        lshaped.ξ = ξ

        init(lshaped)

        return lshaped
    end
end

@traitimpl HasTrustRegion{TrustRegionLShapedSolver}

TrustRegionLShapedSolver(m::JuMPModel,ξ::AbstractVector) = TrustRegionLShapedSolver(m,convert(Vector{Float64},ξ))

function Base.show(io::IO, lshaped::TrustRegionLShapedSolver)
    print(io,"TrustRegionLShapedSolver")
end

function prepareMaster!(lshaped::TrustRegionLShapedSolver,n::Integer)
    lshaped.masterSolver = LQSolver(lshaped.structuredModel)
    # θs
    for i = 1:n
        addvar!(lshaped.masterSolver.model,-Inf,Inf,1.0)
    end
    lshaped.masterSolver.x = copy(lshaped.x)
    append!(lshaped.masterSolver.x,fill(-Inf,n))

    setTrustRegion!(lshaped)
end

@traitfn function setTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    l = max.(lshaped.structuredModel.colLower, lshaped.ξ-lshaped.Δ)
    append!(l,fill(-Inf,lshaped.nscenarios))
    u = min.(lshaped.structuredModel.colUpper, lshaped.ξ+lshaped.Δ)
    append!(u,fill(Inf,lshaped.nscenarios))
    setvarLB!(lshaped.masterSolver.model,l)
    setvarUB!(lshaped.masterSolver.model,u)
end

@traitfn function takeTrustRegionStep!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(lshaped.θs)
    if lshaped.obj <= lshaped.Q̃ - lshaped.γ*abs(lshaped.Q̃-θ)
        println("Major step")
        lshaped.ξ = copy(lshaped.x)
        lshaped.Q̃ = lshaped.obj
        push!(lshaped.Q̃_hist,lshaped.Q̃)
        enlargeTrustRegion!(lshaped)
        lshaped.nMajorSteps += 1
    else
        println("Minor step")
        reduceTrustRegion!(lshaped)
        lshaped.nMinorSteps += 1
    end
end

@traitfn function enlargeTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(lshaped.θs)
    if abs(lshaped.obj - lshaped.Q̃) <= 0.5*(lshaped.Q̃-θ) && norm(lshaped.ξ-lshaped.x,Inf) - lshaped.Δ <= lshaped.τ
        # Enlarge the trust-region radius
        lshaped.Δ = min(lshaped.Δ̅,2*lshaped.Δ)
        push!(lshaped.Δ_hist,lshaped.Δ)
        setTrustRegion!(lshaped)
        return true
    else
        return false
    end
end

@traitfn function reduceTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(lshaped.θs)
    ρ = min(1,lshaped.Δ)*(lshaped.obj-lshaped.Q̃)/(lshaped.Q̃-θ)
    @show ρ
    if ρ > 0
        lshaped.cΔ += 1
        return false
    elseif ρ > 3 || (lshaped.cΔ >= 3 && 1 < ρ <= 3)
        # Reduce the trust-region radius
        lshaped.cΔ = 0
        lshaped.Δ = (1/min(ρ,4))*lshaped.Δ
        push!(lshaped.Δ_hist,lshaped.Δ)
        setTrustRegion!(lshaped)
        return true
    end
end

@traitfn function removeCuts!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    inactive = find(c->!active(lshaped,c),lshaped.committee)
    diff = length(lshaped.committee) - lshaped.nscenarios
    if isempty(inactive) || diff <= 0
        return false
    end
    if diff <= length(inactive)
        inactive = inactive[1:diff]
    end
    append!(lshaped.inactive,lshaped.committee[inactive])
    deleteat!(lshaped.committee,inactive)
    #delconstrs!(lshaped.masterSolver.model,inactive)
    delconstrs!(lshaped.masterSolver.model,inactive+lshaped.masterProblem.numRows)
    return true
end

function (lshaped::TrustRegionLShapedSolver)()
    println("Starting L-Shaped procedure with trust-region")
    println("======================")
    # Initial solve of subproblems at starting guess
    println("Initial solve of subproblems at starting regularizer")
    updateSubProblems!(lshaped.subProblems,lshaped.ξ)
    map(s->addCut!(lshaped,s(),lshaped.ξ),lshaped.subProblems)
    lshaped.Q̃ = sum(lshaped.subObjectives)
    push!(lshaped.Q̃_hist,lshaped.Q̃)
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    lshaped.status = status(lshaped.masterSolver)
    if lshaped.status == :Infeasible
        println("Master is infeasible, aborting procedure.")
        println("======================")
        return
    end
    updateSolution!(lshaped)

    println("Main loop")
    println("======================")

    while true
        if isempty(lshaped.violating)
            # Resolve all subproblems at the current optimal solution
            resolveSubproblems!(lshaped)
            # Update the objective value
            updateObjectiveValue!(lshaped)

            takeTrustRegionStep!(lshaped)

            c = MathProgBase.SolverInterface.getobj(lshaped.masterSolver.model)
            c[end-lshaped.nscenarios+1:end] = 1.0
            MathProgBase.SolverInterface.setobj!(lshaped.masterSolver.model,c)
        else
            # Add at most L violating constraints
            L = 0
            while !isempty(lshaped.violating) && L < lshaped.nscenarios
                constraint = dequeue!(lshaped.violating)
                if satisfied(lshaped,constraint)
                    push!(lshaped.inactive,constraint)
                    continue
                end
                println("Adding violated constraint to committee")
                push!(lshaped.committee,constraint)
                addconstr!(lshaped.masterSolver.model,lowlevel(constraint)...)
                L += 1
            end
        end

        # Resolve master
        println("Solving master problem")
        lshaped.masterSolver()
        lshaped.status = status(lshaped.masterSolver)
        if lshaped.status == :Infeasible
            println("Master is infeasible, aborting procedure.")
            println("======================")
            return
        end
        # Update master solution
        updateSolution!(lshaped)
        #removeCuts!(lshaped)
        if length(lshaped.violating) <= lshaped.nscenarios
            queueViolated!(lshaped)
        end

        if checkOptimality(lshaped)
            # Optimal
            lshaped.status = :Optimal
            updateStructuredModel!(lshaped)
            println("Optimal!")
            println("Objective value: ", lshaped.Q̃)
            println("======================")
            break
        end
    end
end
