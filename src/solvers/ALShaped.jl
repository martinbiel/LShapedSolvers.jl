@with_kw mutable struct ALShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
end

@with_kw struct ALShapedParameters{T <: Real}
    κ::T = 0.3
    τ::T = 1e-6
end

struct ALShaped{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::ALShapedData{T}

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

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::ALShapedParameters{T}

    @implement_trait ALShaped IsParallel

    function (::Type{ALShaped})(model::JuMP.Model,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return LShaped(model,x₀,mastersolver,subsolver)
        end
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(x₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(x₀))
        A = typeof(x₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        lshaped = new{T,A,M,S}(model,
                               ALShapedData{T}(),
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
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               ALShapedParameters{T}(;kw...))
        push!(lshaped.subobjectives,zeros(n))
        push!(lshaped.finished,0)
        push!(lshaped.Q_history,Inf)
        push!(lshaped.θ_history,-Inf)

        init!(lshaped,subsolver)

        return lshaped
    end
end
ALShaped(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = ALShaped(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::ALShaped{T,A,M,S})() where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
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
            end
        end

        # Resolve master
        t = lshaped.solverdata.timestamp
        if lshaped.finished[t] >= lshaped.parameters.κ*lshaped.nscenarios && length(lshaped.cuts) >= lshaped.nscenarios
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

            lshaped.θ_history[t] = θ
            if check_optimality(lshaped)
                # Optimal
                map(w->put!(w,-1),lshaped.work)
                lshaped.solverdata.Q = calculateObjective(lshaped,lshaped.x)
                lshaped.Q_history[t] = lshaped.solverdata.Q
                @async begin
                    close(lshaped.cutqueue)
                    map(wait,finished_workers)
                    println("Workers have finished")
                end
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
            push!(lshaped.θ_history,lshaped.solverdata.θ)
            push!(lshaped.subobjectives,zeros(lshaped.nscenarios))
            push!(lshaped.finished,0)
        end
    end
end
