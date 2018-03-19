# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} = begin
    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,!IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        # Prepare the subproblems
        m = lshaped.structuredmodel
        for i = 1:lshaped.nscenarios
            y₀ = convert(A,rand(subproblem(m,i).numCols))
            push!(lshaped.subproblems,SubProblem(subproblem(m,i),m,i,probability(m,i),copy(lshaped.x),y₀,subsolver))
        end
        lshaped
    end

    function init_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver,IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
        @unpack σ = lshaped.parameters
        # Partitioning
        (jobLength,extra) = divrem(lshaped.nscenarios,nworkers())
        # One extra to guarantee coverage
        if extra > 0
            jobLength += 1
        end
        # Create subproblems on worker processes
        m = lshaped.structuredmodel
        start = 1
        stop = jobLength
        @sync for w in workers()
            lshaped.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            lshaped.mastercolumns[w-1] = RemoteChannel(() -> Channel{Tuple{Int,A}}(round(Int,10/σ)), w)
            put!(lshaped.mastercolumns[w-1],(1,lshaped.x))
            submodels = [subproblem(m,i) for i = start:stop]
            πs = [probability(m,i) for i = start:stop]
            @spawnat w init_subworker!(lshaped.subworkers[w-1],
                                       m,
                                       submodels,
                                       πs,
                                       lshaped.x,
                                       subsolver,
                                       collect(start:stop))
            if start > lshaped.nscenarios
                continue
            end
            start += jobLength
            stop += jobLength
            stop = min(stop,lshaped.nscenarios)
        end
        lshaped
    end
end

@define_traitfn IsParallel calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector) = begin
    function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector,NullTrait)
        return lshaped.c⋅x + sum([subproblem.π*subproblem(x) for subproblem in lshaped.subproblems])
    end

    function calculateObjective(lshaped::AbstractLShapedSolver,x::AbstractVector,IsParallel)
        c = lshaped.structuredmodel.obj.aff.coeffs
        objidx = [v.col for v in lshaped.structuredmodel.obj.aff.vars]
        return c⋅x[objidx] + sum(fetch.([@spawnat w calculate_subobjective(worker,x) for (w,worker) in enumerate(lshaped.subworkers)]))
    end
end

# Parallel routines #
# ======================================================================== #
SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
MasterColumn{A} = RemoteChannel{Channel{Tuple{Int,A}}}
QCut{T} = Tuple{Int,T,SparseHyperPlane{T}}
CutQueue{T} = RemoteChannel{Channel{QCut{T}}}

function init_subworker!(subworker::SubWorker{T,A,S},
                         parent::JuMP.Model,
                         submodels::Vector{JuMP.Model},
                         πs::A,
                         x::A,
                         subsolver::AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems = Vector{SubProblem{T,A,S}}(length(ids))
    for (i,id) = enumerate(ids)
        y₀ = convert(A,rand(submodels[i].numCols))
        subproblems[i] = SubProblem(submodels[i],parent,id,πs[i],x,y₀,subsolver)
    end
    put!(subworker,subproblems)
end

function work_on_subproblems!(subworker::SubWorker{T,A,S},
                              cuts::CutQueue{T},
                              rx::MasterColumn{A}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    while true
        t,x::A = take!(rx)
        if isempty(x)
            println("Worker finished")
            return
        end
        update_subproblems!(subproblems,x)
        for subproblem in subproblems
            println("Solving subproblem: ",subproblem.id)
            cut = subproblem()
            Q::T = cut(x)
            put!(cuts,(t,Q,cut))
            println("Subproblem: ",subproblem.id," solved")
        end
    end
end

function calculate_subobjective(subworker::SubWorker{T,A,S},
                                x::A) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*subproblem(x) for subproblem in subproblems])
    else
        return zero(T)
    end
end
