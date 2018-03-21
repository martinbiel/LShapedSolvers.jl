@with_kw mutable struct ARegularizedData{T <: Real}
    Q::T = 1e10
    Q̃::T = 1e10
    θ::T = -1e10
    σ::T = 1.0
    exact_steps::Int = 0
    approximate_steps::Int = 0
    null_steps::Int = 0
    timestamp::Int = 1
end

@with_kw struct ARegularizedParameters{T <: Real}
    σ::T = 0.4
    τ::T = 1e-6
    γ::T = 0.9
    σ::T = 1.0
    σ̅::T = 4.0
    σ̲::T = 0.5
end

struct ARegularized{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::ARegularizedData{T}

    # Master
    mastersolver::M
    c::A
    x::A
    Q_history::A

    # Subproblems
    nscenarios::Int
    subobjectives::Vector{A}
    finished::Vector{Int}

    # Workers
    subworkers::Vector{SubWorker{T,A,S}}
    work::Vector{Work}
    decisions::Decisions{A}
    cutqueue::CutQueue{T}

    # Trust region
    ξ::A
    Q̃_history::A
    Δ_history::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::ARegularizedParameters{T}

    @implement_trait ARegularized IsRegularized
    @implement_trait ARegularized IsParallel

    function (::Type{ARegularized})(model::JuMP.Model,ξ₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return Regularized(model,ξ₀,mastersolver,subsolver)
        end
        length(ξ₀) != model.numCols && error("Incorrect length of starting guess, has ",length(ξ₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(ξ₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(ξ₀))
        ξ₀_ = convert(AbstractVector{T},copy(ξ₀))
        A = typeof(x₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               ARegularizedData{T}(),
                               msolver,
                               c_,
                               x₀_,
                               A(),
                               n,
                               Vector{A}(),
                               Vector{Int}(),
                               Vector{SubWorker{T,A,S}}(nworkers()),
                               Vector{Work}(nworkers()),
                               RemoteChannel(() -> DecisionChannel(Dict{Int,A}())),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               ξ₀_,
                               A(),
                               A(),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               ARegularizedParameters{T}(;kw...))
        push!(lshaped.subobjectives,zeros(n))
        push!(lshaped.finished,0)
        push!(lshaped.Q_history,Inf)
        push!(lshaped.Q̃_history,Inf)
        push!(lshaped.θ_history,-Inf)

        init!(lshaped,subsolver)

        return lshaped
    end
end
ARegularized(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = ARegularized(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::ARegularized{T,A,M,S})() where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    println("Starting parallel L-Shaped procedure\n")
    println("======================")

    # Start workers
    finished_workers = Vector{Future}(nworkers())
    println("Start workers")
    for w in workers()
        println("Send work to ",w)
        finished_workers[w-1] = remotecall(work_on_subproblems!,
                                           w,
                                           lshaped.subworkers[w-1],
                                           lshaped.work[w-1],
                                           lshaped.cutqueue,
                                           lshaped.decisions)
        println("Now ",w, " is working")
    end
    println("Main loop")
    println("======================")
    while true
        wait(lshaped.cutqueue)
        while isready(lshaped.cutqueue)
            println("Cuts are ready")
            # Add new cuts from subworkers
            t::Int,Q::T,cut::SparseHyperPlane{T} = take!(lshaped.cutqueue)
            if !bounded(cut)
                println("Subproblem ",cut.id," is unbounded, aborting procedure.")
                println("======================")
                return
            end
            addcut!(lshaped,cut,Q)
            lshaped.subobjectives[t][cut.id] = Q
            lshaped.finished[t] += 1
            if lshaped.finished[t] == lshaped.nscenarios
                lshaped.Q_history[t] = calculate_objective_value(lshaped,lshaped.subobjectives[t])
                if lshaped.Q_history[t] <= lshaped.solverdata.Q
                    lshaped.solverdata.Q = lshaped.Q_history[t]
                end
                lshaped.x[:] = fetch(lshaped.decisions,t)
                take_step!(lshaped)
            end
        end

        # Resolve master
        if lshaped.finished[lshaped.solverdata.timestamp] >= lshaped.parameters.σ*lshaped.nscenarios && length(lshaped.cuts) >= lshaped.nscenarios
            # Update the optimization vector
            println("Solving master problem")
            lshaped.mastersolver(lshaped.x)
            if status(lshaped.mastersolver) == :Infeasible
                println("Master is infeasible, aborting procedure.")
                println("======================")
                return
            end

            # Update master solution
            update_solution!(lshaped)

            θ = calculate_estimate(lshaped)
            lshaped.solverdata.θ = θ
            t = lshaped.solverdata.timestamp
            lshaped.Q̃_history[t] = lshaped.solverdata.Q̃
            lshaped.Δ_history[t] = lshaped.solverdata.Δ
            lshaped.θ_history[t] = θ

            if check_optimality(lshaped)
                # Optimal
                map(w->put!(w,-1),lshaped.work)
                lshaped.x[:] = lshaped.ξ[:]
                lshaped.solverdata.Q = calculateObjective(lshaped,lshaped.x)
                lshaped.Q_history[t] = lshaped.solverdata.Q
                map(wait,finished_workers)
                println("Optimal!")
                println("Objective value: ", lshaped.Q_history[t])
                println("======================")
                return
            end

            # Send new decision vector to workers
            put!(lshaped.decisions,t+1,lshaped.x)
            for w in lshaped.work
                put!(w,t+1)
            end

            # Prepare memory for next timestamp
            lshaped.solverdata.timestamp += 1
            push!(lshaped.Q_history,lshaped.solverdata.Q)
            push!(lshaped.Q̃_history,lshaped.solverdata.Q̃)
            push!(lshaped.Δ_history,lshaped.solverdata.Δ)
            push!(lshaped.θ_history,lshaped.solverdata.θ)
            push!(lshaped.subobjectives,zeros(lshaped.nscenarios))
            push!(lshaped.finished,0)
        end
    end
end
