abstract type AbstractLShapedSolver{T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver} <: AbstractStructuredModel end

nscenarios(lshaped::AbstractLShapedSolver) = lshaped.nscenarios

# Initialization #
# ======================================================================== #
function init!(lshaped::AbstractLShapedSolver{T,A,M,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Initialize progress meter
    lshaped.progress.thresh = lshaped.parameters.τ
    # Prepare the master optimization problem
    prepare_master!(lshaped)
    # Finish initialization based on solver traits
    init_subproblems!(lshaped,subsolver)
    init_solver!(lshaped)
end
# ======================================================================== #

# Functions #
# ======================================================================== #
function set_params!(lshaped::AbstractLShapedSolver; kwargs...)
    for (k,v) in kwargs
        setfield!(lshaped.parameters,k,v)
    end
end

function update_solution!(lshaped::AbstractLShapedSolver)
    ncols = lshaped.structuredmodel.numCols
    x = getsolution(lshaped.mastersolver)
    lshaped.mastervector[:] = x
    lshaped.x[1:ncols] = x[1:ncols]
    lshaped.θs[:] = x[end-lshaped.nscenarios+1:end]
    nothing
end

function update_structuredmodel!(lshaped::AbstractLShapedSolver)
    lshaped.structuredmodel.colVal = copy(lshaped.x)
    lshaped.structuredmodel.objVal = lshaped.c⋅lshaped.x + sum(lshaped.subobjectives)

    for i in 1:lshaped.nscenarios
        m = subproblem(lshaped.structuredmodel,i)
        m.colVal = copy(getsolution(lshaped.subproblems[i].solver))
        m.objVal = getobjval(lshaped.subproblems[i].solver)
    end
    nothing
end

function calculate_estimate(lshaped::AbstractLShapedSolver)
    return lshaped.c⋅lshaped.x + sum(lshaped.θs)
end

function current_objective_value(lshaped::AbstractLShapedSolver,Qs::AbstractVector)
    return lshaped.c⋅lshaped.x + sum(Qs)
end
current_objective_value(lshaped) = current_objective_value(lshaped,lshaped.subobjectives)

function get_solution(lshaped::AbstractLShapedSolver)
    return lshaped.x
end

function get_objective_value(lshaped::AbstractLShapedSolver)
    if !isempty(lshaped.Q_history)
        return lshaped.Q_history[end]
    else
        return calculate_objective_value(lshaped)
    end
end

function prepare_master!(lshaped::AbstractLShapedSolver)
    # θs
    for i = 1:lshaped.nscenarios
        addvar!(lshaped.mastersolver.lqmodel,-Inf,Inf,1.0)
        push!(lshaped.mastervector,-1e10)
    end
end

function resolve_subproblems!(lshaped::AbstractLShapedSolver{T,A,M,S}) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    # Update subproblems
    update_subproblems!(lshaped.subproblems,lshaped.x)

    # Solve sub problems
    for subproblem ∈ lshaped.subproblems
        cut::SparseHyperPlane{T} = subproblem()
        if !bounded(cut)
            warn("Subproblem ",cut.id," is unbounded, aborting procedure.")
            return -Inf
        end
        addcut!(lshaped,cut)
        lshaped.subobjectives[cut.id] = cut(lshaped.x)
    end

    # Return current objective value
    return current_objective_value(lshaped)
end

function iterate_nominal!(lshaped::AbstractLShapedSolver)
    # Resolve all subproblems at the current optimal solution
    Q = resolve_subproblems!(lshaped)
    if Q == -Inf
        return :Unbounded
    end
    lshaped.solverdata.Q = Q
    # Update the optimization vector
    take_step!(lshaped)
    # Resolve master
    try
        lshaped.mastersolver(lshaped.mastervector)
    catch
        # Master problem could not be solved for some reason.
        @unpack Q,θ = lshaped.solverdata
        gap = abs(θ-Q)/(abs(Q)+1e-10)
        warn("Master problem could not be solved, solver returned status $(status(lshaped.mastersolver)). The following relative tolerance was reached: $(@sprintf("%.1e",gap)). Aborting procedure.")
        return :StoppedPrematurely
    end
    if status(lshaped.mastersolver) == :Infeasible
        warn("Master is infeasible. Aborting procedure.")
        return :Infeasible
    end
    # Update master solution
    update_solution!(lshaped)
    lshaped.solverdata.θ = calculate_estimate(lshaped)
    # Log progress
    log!(lshaped)
    # Check optimality
    if check_optimality(lshaped)
        # Optimal
        lshaped.solverdata.Q = calculate_objective_value(lshaped,lshaped.x)
        push!(lshaped.Q_history,lshaped.solverdata.Q)
        return :Optimal
    end
    # Project (if applicable)
    project!(lshaped)
    # Just return a valid status for this iteration
    return :Valid
end

function log!(lshaped::AbstractLShapedSolver)
    @unpack Q,θ = lshaped.solverdata
    push!(lshaped.Q_history,Q)
    push!(lshaped.θ_history,θ)
    lshaped.solverdata.iterations += 1

    log_regularization!(lshaped)

    if lshaped.parameters.log
        current_gap = gap(lshaped)
        ProgressMeter.update!(lshaped.progress,current_gap,
                              showvalues = [
                                  ("Objective",Q),
                                  ("Gap",current_gap),
                                  ("Number of cuts",length(lshaped.cuts))
                              ])
    end
end

function log!(lshaped::AbstractLShapedSolver,t::Integer)
    @unpack Q,θ = lshaped.solverdata
    lshaped.Q_history[t] = Q
    lshaped.θ_history[t] = θ

    log_regularization!(lshaped,t)
end

function check_optimality(lshaped::AbstractLShapedSolver)
    @unpack τ = lshaped.parameters
    @unpack θ = lshaped.solverdata
    return θ > -Inf && gap(lshaped) <= τ
end
# ======================================================================== #

# Cut functions #
# ======================================================================== #
active(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = active(hyperplane,lshaped.x,lshaped.parameters.τ)
active(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = optimal(cut,lshaped.x,lshaped.θs[cut.id],lshaped.parameters.τ)
satisfied(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = satisfied(hyperplane,lshaped.x,lshaped.parameters.τ)
satisfied(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = satisfied(cut,lshaped.x,lshaped.θs[cut.id],lshaped.parameters.τ)
violated(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = !satisfied(lshaped,hyperplane)
gap(lshaped::AbstractLShapedSolver,hyperplane::HyperPlane) = gap(hyperplane,lshaped.x)
gap(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = gap(cut,lshaped.x,lshaped.θs[cut.id])

function addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut},Q::Real)
    θ = lshaped.θs[cut.id]
    @unpack τ = lshaped.parameters

    if θ > -Inf && abs(θ-Q) <= τ*(1+abs(Q))
        # Optimal with respect to this subproblem
        return false
    end

    process_cut!(lshaped,cut)
    addconstr!(lshaped.mastersolver.lqmodel,lowlevel(cut)...)
    push!(lshaped.cuts,cut)
    return true
end
addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut},x::AbstractVector) = addcut!(lshaped,cut,cut(x))
addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut}) = addcut!(lshaped,cut,lshaped.x)

function addcut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{FeasibilityCut})
    process_cut!(lshaped,cut)
    addconstr!(lshaped.mastersolver.lqmodel,lowlevel(cut)...)
    push!(lshaped.cuts,cut)
    return true
end

function show(io::IO, lshaped::AbstractLShapedSolver)
    println(io,typeof(lshaped).name.name)
    println(io,"State:")
    show(io,lshaped.solverdata)
    println(io,"Parameters:")
    show(io,lshaped.parameters)
end

function show(io::IO, ::MIME"text/plain", lshaped::AbstractLShapedSolver)
    show(io,lshaped)
end
# ======================================================================== #

# Plot recipe #
# ======================================================================== #
@recipe f(lshaped::AbstractLShapedSolver) = lshaped,-1
@recipe function f(lshaped::AbstractLShapedSolver, time::Real; showθ = false)
    length(lshaped.Q_history) > 0 || error("No solution data. Has solver been run?")
    showθ && (length(lshaped.θ_history) > 0 || error("No solution data. Has solver been run?"))
    Qmin = showθ ? minimum(lshaped.θ_history) : minimum(lshaped.Q_history)
    Qmax = maximum(lshaped.Q_history)
    increment = std(lshaped.Q_history)

    linewidth --> 4
    linecolor --> :black
    tickfontsize := 14
    tickfontfamily := "sans-serif"
    guidefontsize := 16
    guidefontfamily := "sans-serif"
    titlefontsize := 22
    titlefontfamily := "sans-serif"
    xlabel := time == -1 ? "Iteration" : "Time [s]"
    ylabel := "Q"
    ylims --> (Qmin-increment,Qmax+increment)
    if time == -1
        xlims --> (1,length(lshaped.Q_history)+1)
        xticks --> 1:5:length(lshaped.Q_history)
    else
        xlims --> (0,time)
        xticks --> linspace(0,time,ceil(Int,length(lshaped.Q_history)/5))
    end
    yticks --> Qmin:increment:Qmax
    xformatter := (d) -> @sprintf("%.1f",d)
    yformatter := (d) -> begin
        if abs(d) <= sqrt(eps())
            "0.0"
        elseif (log10(abs(d)) < -2.0 || log10(abs(d)) > 3.0)
            @sprintf("%.4e",d)
        elseif log10(abs(d)) > 2.0
            @sprintf("%.1f",d)
        else
            @sprintf("%.2f",d)
        end
    end

    @series begin
        label --> "Q"
        seriescolor --> :black
        if time == -1
            1:1:length(lshaped.Q_history),lshaped.Q_history
        else
            linspace(0,time,length(lshaped.Q_history)),lshaped.Q_history
        end
    end

    if showθ
        @series begin
            label --> "θ"
            linestyle --> :dash
            seriescolor --> :red
            linecolor --> :red
            linewidth --> 2
            if time == -1
                1:1:length(lshaped.θ_history),lshaped.θ_history
            else
                linspace(0,time,length(lshaped.θ_history)),lshaped.θ_history
            end
        end
    end
end
# ======================================================================== #
