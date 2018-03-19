@with_kw mutable struct PLShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    timestamp::Int = 1
end

@with_kw struct PLShapedParameters{T <: Real}
    σ::T = 0.4
    τ::T = 1e-6
end

struct PLShaped{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMP.Model
    solverdata::PLShapedData{T}

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
    mastercolumns::Vector{MasterColumn{A}}
    cutqueue::CutQueue{T}

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}
    θ_history::A

    # Params
    parameters::PLShapedParameters{T}

    @implement_trait PLShaped IsParallel

    function (::Type{PLShaped})(model::JuMP.Model,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
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
                               PLShapedData{T}(),
                               msolver,
                               c_,
                               x₀_,
                               A(),
                               n,
                               Vector{A}(),
                               Vector{Int}(),
                               Vector{SubWorker{T,A,S}}(nworkers()),
                               Vector{MasterColumn{A}}(nworkers()),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               A(),
                               PLShapedParameters{T}(;kw...))
        push!(lshaped.subobjectives,zeros(n))
        push!(lshaped.finished,0)
        push!(lshaped.Q_history,Inf)
        init!(lshaped,subsolver)

        return lshaped
    end
end
PLShaped(model::JuMP.Model,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...) = PLShaped(model,rand(model.numCols),mastersolver,subsolver; kw...)

function (lshaped::PLShaped{T,A,M,S})() where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    println("Starting parallel L-Shaped procedure\n")
    println("======================")

    # Start workers
    finished_workers = Vector{Future}(nworkers())
    println("Start workers")
    for w in workers()
        println("Send work to ",w)
        finished_workers[w-1] = @spawnat w work_on_subproblems!(lshaped.subworkers[w-1],
                                                                lshaped.cutqueue,
                                                                lshaped.mastercolumns[w-1])
        println("Now ",w, " is working")
    end
    println("Main loop")
    println("======================")
    while true
        wait(lshaped.cutqueue)
        while isready(lshaped.cutqueue)
            println("Cuts are ready")
            # Add new cuts from subworkers
            t,Q::T,cut::SparseHyperPlane{T} = take!(lshaped.cutqueue)
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

                if check_optimality(lshaped)
                    # Optimal
                    map(rx->put!(rx,(-1,[])),lshaped.mastercolumns)
                    #update_structuredmodel!(lshaped)
                    map(wait,finished_workers)
                    println("Optimal!")
                    println("Objective value: ", lshaped.Q_history[t])
                    println("======================")
                    return nothing
                end
            end
        end

        # Resolve master
        if lshaped.finished[lshaped.solverdata.timestamp] >= lshaped.parameters.σ*lshaped.nscenarios && length(lshaped.cuts) >= lshaped.nscenarios
            println("Solving master problem")
            lshaped.mastersolver(lshaped.x)
            if status(lshaped.mastersolver) == :Infeasible
                println("Master is infeasible, aborting procedure.")
                println("======================")
                return
            end
            # Update master solution
            update_solution!(lshaped)
            lshaped.solverdata.timestamp += 1
            for rx in lshaped.mastercolumns
                put!(rx,(lshaped.solverdata.timestamp,lshaped.x))
            end
            θ = calculate_estimate(lshaped)
            lshaped.solverdata.θ = θ
            push!(lshaped.Q_history,Inf)
            push!(lshaped.θ_history,θ)
            push!(lshaped.subobjectives,zeros(lshaped.nscenarios))
            push!(lshaped.finished,0)
        end
    end
end
