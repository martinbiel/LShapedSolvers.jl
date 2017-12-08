struct LShapedSolver{float_t <: Real, array_t <: AbstractVector, msolver_t <: LQSolver, ssolver_t <: LQSolver} <: AbstractLShapedSolver{float_t,array_t,msolver_t,ssolver_t}
    structuredmodel::JuMPModel

    # Master
    mastersolver::msolver_t
    x::array_t
    obj_hist::array_t

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{float_t,array_t,ssolver_t}}
    subobjectives::array_t

    # Cuts
    θs::array_t
    cuts::Vector{SparseHyperPlane{<:HyperPlaneType,float_t}}

    # Params
    τ::float_t

    function (::Type{LShapedSolver})(model::JuMPModel,x₀::AbstractVector,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver)
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:Stochastic) && error("The provided model is not structured")

        float_t = promote_type(eltype(x₀),Float32)
        x₀_ = convert(AbstractVector{float_t},x₀)
        array_t = typeof(x₀_)

        msolver = LQSolver(model,mastersolver,copy(x₀_))
        msolver_t = typeof(msolver)
        ssolver_t = LQSolver{float_t,array_t,typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        println(ssolver_t)

        lshaped = new{float_t,
                      array_t,
                      msolver_t,
                      ssolver_t}(model,
                                 msolver,
                                 x₀_,
                                 array_t(),
                                 num_scenarios(model),
                                 Vector{SubProblem{float_t,array_t,ssolver_t}}(),
                                 array_t(),
                                 array_t(),
                                 Vector{SparseHyperPlane{<:HyperPlaneType,float_t}}(),
                                 convert(float_t,1e-6))
        init!(lshaped,subsolver)

        return lshaped
    end
end
LShapedSolver(model::JuMPModel,mastersolver::AbstractMathProgSolver,subsolver::AbstractMathProgSolver) = LShapedSolver(model,rand(model.numCols),mastersolver,subsolver)

function (lshaped::LShapedSolver{float_t,array_t,msolver_t,ssolver_t})() where {float_t <: Real, array_t <: AbstractVector, msolver_t <: LQSolver, ssolver_t <: LQSolver}
    println("Starting L-Shaped procedure")
    println("======================")

    println("Main loop")
    println("======================")

    obj = convert(float_t,Inf)

    while true
        # Resolve all subproblems at the current optimal solution
        resolve_subproblems!(lshaped)
        obj = calculate_objective_value(lshaped)
        push!(lshaped.obj_hist,obj)

        if check_optimality(lshaped)
            # Optimal
            update_structuredmodel!(lshaped)
            println("Optimal!")
            println("Objective value: ", sum(lshaped.subobjectives))
            println("======================")
            break
        end

        # Resolve master
        println("Solving master problem")
        lshaped.mastersolver()
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
#     obj_hist::AbstractVector

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
#         push!(lshaped.obj_hist,lshaped.obj)
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
