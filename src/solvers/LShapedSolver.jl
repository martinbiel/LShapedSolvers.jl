struct LShapedSolver{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractLShapedSolver{T,A,M,S}
    structuredmodel::JuMPModel

    # Master
    mastersolver::M
    x::A
    objhistory::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}
    subobjectives::A

    # Cuts
    θs::A
    cuts::Vector{SparseHyperPlane{T}}

    # Params
    τ::T

    function (::Type{LShapedSolver})(model::JuMPModel,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver)
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:Stochastic) && error("The provided model is not structured")

        T = promote_type(eltype(x₀),Float32)
        x₀_ = convert(AbstractVector{T},x₀)
        A = typeof(x₀_)

        msolver = LQSolver(model,mastersolver)
        M = typeof(msolver)
        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}

        lshaped = new{T,A,M,S}(model,
                               msolver,
                               x₀_,
                               A(),
                               num_scenarios(model),
                               Vector{SubProblem{T,A,S}}(),
                               A(zeros(num_scenarios(model))),
                               A(fill(-Inf,num_scenarios(model))),
                               Vector{SparseHyperPlane{T}}(),
                               convert(T,1e-6))
        init!(lshaped,subsolver)

        return lshaped
    end
end
LShapedSolver(model::JuMPModel,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver) = LShapedSolver(model,rand(model.numCols),mastersolver,subsolver)

function (lshaped::LShapedSolver{T,A,M,S})() where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    println("Starting L-Shaped procedure")
    println("======================")

    println("Main loop")
    println("======================")

    obj = convert(T,Inf)

    while true
        # Resolve all subproblems at the current optimal solution
        resolve_subproblems!(lshaped)
        obj = calculate_objective_value(lshaped)
        push!(lshaped.objhistory,obj)

        if check_optimality(lshaped)
            # Optimal
            update_structuredmodel!(lshaped)
            println("Optimal!")
            println("Objective value: ", calculate_objective_value(lshaped))
            println("======================")
            break
        end

        # Resolve master
        println("Solving master problem")
        lshaped.mastersolver(lshaped.x)
        if status(lshaped.mastersolver) == :Infeasible
            println("Master is infeasible, aborting procedure.")
            println("======================")
            return
        end
        # Update master solution
        update_solution!(lshaped)
    end
end

# mutable struct AsynchronousLShapedSolver <: AbstractLShapedSolver
#     structuredModel::JuMPModel

#     # Master
#     masterSolver::AbstractLQSolver
#     x::AbstractVector
#     obj::Real
#     objhistory::AbstractVector

#     # Subproblems
#     nscenarios::Integer
#     subObjectives::AbstractVector

#     # Workers
#     subworkers::Vector{RemoteChannel}
#     masterColumns::Vector{RemoteChannel}
#     cutQueue::RemoteChannel
#     finished::RemoteChannel

#     # Cuts
#     θs::AbstractVector
#     cuts::Vector{HyperPlane}
#     nOptimalityCuts::Integer
#     nFeasibilityCuts::Integer

#     # Params
#     status::Symbol
#     τ::Float64

#     function AsynchronousLShapedSolver(m::JuMPModel,x₀::AbstractVector)
#         if nworkers() == 1
#             warn("There are no worker processes, defaulting to serial version of algorithm")
#             return LShapedSolver(m,x₀)
#         end
#         lshaped = new(m)

#         if length(x₀) != m.numCols
#             throw(ArgumentError(string("Incorrect length of starting guess, has ",length(x₀)," should be ",m.numCols)))
#         end

#         lshaped.x = x₀
#         init(lshaped)

#         return lshaped
#     end
# end
# AsynchronousLShapedSolver(m::JuMPModel) = AsynchronousLShapedSolver(m,rand(m.numCols))

# @implement_trait AsynchronousLShapedSolver IsParallel

# function Base.show(io::IO, lshaped::AsynchronousLShapedSolver)
#     print(io,"AsynchronousLShapedSolver")
# end

# function (lshaped::AsynchronousLShapedSolver)()
#     println("Starting asynchronous L-Shaped procedure\n")
#     println("======================")

#     finished_workers = Vector{Future}(nworkers())
#     println("Start workers")
#     for w in workers()
#         println("Send work to ",w)
#         finished_workers[w-1] = @spawnat w work_on_subproblems(lshaped.subworkers[w-1],lshaped.cutQueue,lshaped.masterColumns[w-1])
#         println("Now ",w, " is working")
#     end
#     println("Main loop")
#     println("======================")
#     while true
#         wait(lshaped.cutQueue)
#         println("Cuts are ready")
#         while isready(lshaped.cutQueue)
#             # Add new cuts from subworkers
#             cut = take!(lshaped.cutQueue)
#             if !bounded(cut)
#                 println("Subproblem ",cut.id," is unbounded, aborting procedure.")
#                 println("======================")
#                 return
#             end
#             addCut!(lshaped,cut)
#         end

#         # Resolve master
#         println("Solving master problem")
#         lshaped.masterSolver()
#         lshaped.status = status(lshaped.masterSolver)
#         if lshaped.status == :Infeasible
#             println("Master is infeasible, aborting procedure.")
#             println("======================")
#             return
#         end
#         # Update master solution
#         updateSolution!(lshaped)
#         updateObjectiveValue!(lshaped)
#         push!(lshaped.objhistory,lshaped.obj)
#         for rx in lshaped.masterColumns
#             put!(rx,lshaped.x)
#         end

#         if checkOptimality(lshaped)
#             # Optimal
#             lshaped.status = :Optimal
#             map(rx->put!(rx,[]),lshaped.masterColumns)
#             map(wait,finished_workers)
#             println("Optimal!")
#             println("Objective value: ", sum(lshaped.subObjectives))
#             println("======================")
#             break
#         end
#     end
# end
