@with_kw mutable struct TrustRegionData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    Δ::T = 1.0
    cΔ::Int = 0
    major_steps::Int = 0
    minor_steps::Int = 0
end

@with_kw struct TrustRegionParameters{T <: Real}
    τ::T = 1e-6
    γ::T = 1e-4
    Δ = 1.0
    Δ̅::T = 1.0
    log::Bool = true
end

struct TrustRegion{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::TrustRegionData{T}

    # Master
    mastersolver::M
    c::A
    x::A

    committee::Vector{SparseHyperPlane{T}}
    inactive::Vector{SparseHyperPlane{T}}
    violating::PriorityQueue{SparseHyperPlane{T},T}

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}
    subobjectives::A

    # Trust region
    ξ::A
    Q_history::A
    Q̃_history::A
    Δ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::TrustRegionParameters{T}
    progress::ProgressThresh{T}

    @implement_trait TrustRegion HasTrustRegion

    function (::Type{TrustRegion})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() > 1
            warn("There are worker processes, consider using distributed version of algorithm")
        end
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(ξ₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               TrustRegionData{T}(),
                               msolver,
                               c_,
                               x₀_,
                               convert(Vector{SparseHyperPlane{T}},linearconstraints(model)),
                               Vector{SparseHyperPlane{T}}(),
                               PriorityQueue{SparseHyperPlane{T},T}(Reverse),
                               n,
                               Vector{SubProblem{T,A,S}}(),
                               A(zeros(n)),
                               ξ₀_,
                               A(),
                               A(),
                               A(),
                               A(fill(-1e10,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               TrustRegionParameters{T}(;kw...),
                               ProgressThresh(1.0, "TR L-Shaped Gap "))
        lshaped.progress.thresh = lshaped.parameters.τ
        init!(lshaped,subsolver)

        return lshaped
    end
end
TrustRegion(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = TrustRegion(model,rand(model.numCols),mastersolver,subsolver;kw...)

function (lshaped::TrustRegion)()
    # Reset timer
    lshaped.progress.tfirst = lshaped.progress.tlast = time()
    # Start procedure
    while true
        status = iterate!(lshaped)
        if status != :Valid
            return status
        end

        if check_optimality(lshaped)
            # Optimal
            lshaped.x[:] = lshaped.ξ[:]
            lshaped.solverdata.Q = calculate_objective_value(lshaped,lshaped.x)
            push!(lshaped.Q_history,lshaped.solverdata.Q)
            return :Optimal
        end
    end
end

function iterate!(lshaped::TrustRegion)
    if isempty(lshaped.violating)
        # Resolve all subproblems at the current optimal solution
        lshaped.solverdata.Q = resolve_subproblems!(lshaped)
        if lshaped.solverdata.Q == -Inf
            return :Unbounded
        end
        # Update the optimization vector
        take_step!(lshaped)
    else
        # Add at most L violating constraints
        # L = 0
        # while !isempty(lshaped.violating) && L < lshaped.nscenarios
        #     constraint = dequeue!(lshaped.violating)
        #     if satisfied(lshaped,constraint)
        #         push!(lshaped.inactive,constraint)
        #         continue
        #     end
        #     push!(lshaped.committee,constraint)
        #     addconstr!(lshaped.mastersolver.lqmodel,lowlevel(constraint)...)
        #     L += 1
        # end
    end
    # Resolve master
    lshaped.mastersolver(lshaped.x)
    if status(lshaped.mastersolver) == :Infeasible
        warn("Master is infeasible, aborting procedure.")
        return :Infeasible
    end
    # Update master solution
    update_solution!(lshaped)
    lshaped.solverdata.θ = calculate_estimate(lshaped)
    # remove_inactive!(lshaped)
    # if length(lshaped.violating) <= lshaped.nscenarios
    #     queueViolated!(lshaped)
    # end
    @unpack Q,Q̃,θ = lshaped.solverdata
    push!(lshaped.Q_history,Q)
    push!(lshaped.Δ_history,lshaped.solverdata.Δ)
    push!(lshaped.Q̃_history,Q̃)
    push!(lshaped.θ_history,θ)
    gap = abs(θ-Q)/(1+abs(Q))
    if lshaped.parameters.log
        ProgressMeter.update!(lshaped.progress,gap,
                              showvalues = [
                                  ("Objective",Q),
                                  ("Gap",gap),
                                  ("Number of cuts",length(lshaped.cuts))
                              ])
    end
    return :Valid
end
