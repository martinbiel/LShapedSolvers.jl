mutable struct TrustRegionLShapedSolver <: AbstractLShapedSolver
    structuredModel::JuMPModel

    masterModel::JuMPModel
    masterProblem::LPProblem
    masterSolver::LPSolver
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
    Δ::Real
    Δ̅::Real
    cΔ::Integer
    nMajorSteps::Integer
    nMinorSteps::Integer

    # Cuts
    θs
    ready::BitArray
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

        lshaped.ξ = ξ
        lshaped.Δ = max(1.0,0.2*norm(ξ,Inf))
        lshaped.Δ̅ = 1000*lshaped.Δ
        lshaped.cΔ = 0
        lshaped.γ = 1e-4
        lshaped.nMajorSteps = 0
        lshaped.nMinorSteps = 0

        init(lshaped)

        lshaped.committee = Vector{AbstractHyperplane}()
        #lshaped.committee = linearconstraints(lshaped.structuredModel)
        lshaped.inactive = Vector{AbstractHyperplane}()
        lshaped.violating = PriorityQueue(Reverse)

        lshaped.Q̃ = calculateObjective(lshaped,ξ)

        return lshaped
    end
end

@traitimpl HasTrustRegion{TrustRegionLShapedSolver}

TrustRegionLShapedSolver(m::JuMPModel,ξ::AbstractVector) = TrustRegionLShapedSolver(m,convert(Vector{Float64},ξ))

function Base.show(io::IO, lshaped::TrustRegionLShapedSolver)
    print(io,"TrustRegionLShapedSolver")
end

function prepareMaster!(lshaped::TrustRegionLShapedSolver,n::Integer)
    lshaped.θs = @variable(lshaped.masterModel,θ[i = 1:n],start=-Inf)
    c = lshaped.structuredModel.obj.aff.coeffs
    c *= lshaped.structuredModel.objSense == :Min ? 1 : -1
    objidx = [v.col for v in lshaped.structuredModel.obj.aff.vars]
    x = [Variable(lshaped.masterModel,i) for i in 1:(lshaped.structuredModel.numCols)]

    @objective(lshaped.masterModel,Min,sum(c.*x[objidx]) + sum(lshaped.θs))
    lshaped.ready = falses(n)

    p = LPProblem(lshaped.masterModel)
    lshaped.masterProblem = p
    lshaped.masterSolver = LPSolver(p)
    setTrustRegion!(lshaped)
end

@traitfn function setTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    l = lshaped.structuredModel.colLower
    u = lshaped.structuredModel.colUpper
    for i = 1:lshaped.structuredModel.numCols
        lshaped.masterProblem.l[i] = max(l[i],lshaped.ξ[i]-lshaped.Δ)
        lshaped.masterProblem.u[i] = min(u[i],lshaped.ξ[i]+lshaped.Δ)
    end
    setvarLB!(lshaped.masterSolver.model, lshaped.masterProblem.l)
    setvarUB!(lshaped.masterSolver.model, lshaped.masterProblem.u)
end

@traitfn function takeTrustRegionStep!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(getvalue(lshaped.θs))
    if lshaped.obj <= lshaped.Q̃ - lshaped.γ*abs(lshaped.Q̃-θ)
        println("Major step")
        lshaped.ξ = copy(lshaped.x)
        lshaped.Q̃ = lshaped.obj
        enlargeTrustRegion!(lshaped)
        lshaped.nMajorSteps += 1
    else
        println("Minor step")
        reduceTrustRegion!(lshaped)
        lshaped.nMinorSteps += 1
    end
end

@traitfn function enlargeTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(getvalue(lshaped.θs))
    if abs(lshaped.obj - lshaped.Q̃) <= 0.5*(lshaped.Q̃-θ) && norm(lshaped.ξ-lshaped.x,Inf) - lshaped.Δ <= lshaped.τ
        # Enlarge the trust-region radius
        lshaped.Δ = min(lshaped.Δ̅,2*lshaped.Δ)
        setTrustRegion!(lshaped)
        return true
    else
        return false
    end
end

@traitfn function reduceTrustRegion!{LS <: AbstractLShapedSolver; HasTrustRegion{LS}}(lshaped::LS)
    θ = sum(getvalue(lshaped.θs))
    ρ = min(1,lshaped.Δ)*(lshaped.obj-lshaped.Q̃)/(lshaped.Q̃-θ)
    @show ρ
    if ρ > 0
        lshaped.cΔ += 1
        return false
    elseif ρ > 3 || (lshaped.cΔ >= 3 && 1 < ρ <= 3)
        # Reduce the trust-region radius
        lshaped.cΔ = 0
        lshaped.Δ = (1/min(ρ,4))*lshaped.Δ
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
    println("Starting trust-region L-Shaped procedure\n")
    println("======================")
    updateSubProblems!(lshaped.subProblems,lshaped.ξ)
    map(s->addCut!(lshaped,s()),lshaped.subProblems)
    # Initial solve of master problem
    println("Initial solve of master")
    lshaped.masterSolver()
    updateSolution(lshaped.masterSolver,lshaped.masterModel)
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
        updateSolution(lshaped.masterSolver,lshaped.masterModel)

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
            println("======================")
            break
        end
    end
end
