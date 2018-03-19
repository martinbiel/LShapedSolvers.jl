@with_kw mutable struct PLShapedData{T <: Real}
    Q::T = 1e10
    θ::T = -1e10
    iterations::Int = 0
end

@with_kw struct PLShapedParameters{T <: Real}
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
    θ_history::A

    # Subproblems
    nscenarios::Int
    subobjectives::A

    # Workers
    subworkers::Vector{SubWorker{T,A,S}}
    mastercolumns::Vector{MasterColumn{A}}
    cutqueue::CutQueue{T}

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}

    # Params
    parameters::PLShapedParameters{T}

    @implement_trait PLShaped IsParallel

    function (::Type{PLShaped})(model::JuMP.Model,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() == 1
            warn("There are no worker processes, defaulting to serial version of algorithm")
            return LShapedSolver(model,x₀,mastersolver,subsolver)
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
                               A(),
                               n,
                               A(zeros(n)),
                               Vector{SubWorker{T,A,S}}(nworkers()),
                               Vector{MasterColumn{A}}(nworkers()),
                               RemoteChannel(() -> Channel{QCut{T}}(4*nworkers()*n)),
                               A(fill(-Inf,n)),
                               Vector{SparseHyperPlane{T}}(),
                               PLShapedParameters{T}(;kw...))
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
    tic()
    while true
        wait(lshaped.cutqueue)
        while isready(lshaped.cutqueue)
            println("Cuts are ready")
            # Add new cuts from subworkers
            Q::T,cut::SparseHyperPlane{T} = take!(lshaped.cutqueue)
            if !bounded(cut)
                println("Subproblem ",cut.id," is unbounded, aborting procedure.")
                println("======================")
                return
            end
            addcut!(lshaped,cut,Q)
        end

        if check_optimality(lshaped)
            # Optimal
            map(rx->put!(rx,[]),lshaped.mastercolumns)
            #update_structuredmodel!(lshaped)
            toc()
            map(wait,finished_workers)
            println("Optimal!")
            println("Objective value: ", calculate_objective_value(lshaped))
            println("======================")
            break
        end

        # Resolve master
        if length(lshaped.cuts) >= lshaped.nscenarios
            println("Solving master problem")
            lshaped.mastersolver(lshaped.x)
            if status(lshaped.mastersolver) == :Infeasible
                println("Master is infeasible, aborting procedure.")
                println("======================")
                return
            end
            # Update master solution
            update_solution!(lshaped)
            push!(lshaped.Q_history,calculate_objective_value(lshaped))
            push!(lshaped.θ_history,calculate_estimate(lshaped))
            for rx in lshaped.mastercolumns
                put!(rx,lshaped.x)
            end
        end
    end
end
