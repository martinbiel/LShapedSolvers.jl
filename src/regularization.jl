# ------------------------------------------------------------
# UsesRegularization: Algorithm uses some regularization method
# ------------------------------------------------------------
@define_trait UsesRegularization = begin
    IsRegularized  # Algorithm uses the regularized decomposition method of Ruszczyński
    HasTrustRegion # Algorithm uses the trust-region method of Linderoth/Wright
    HasLevels      # Algorithm uses the level set method of Lemarcheral
end

@define_traitfn UsesRegularization init_solver!(lshaped::AbstractLShapedSolver) = begin
    function init_solver!(lshaped::AbstractLShapedSolver,!UsesRegularization)
        nothing
    end
end

@define_traitfn UsesRegularization log_regularization!(lshaped::AbstractLShapedSolver) = begin
    function log_regularization!(lshaped::AbstractLShapedSolver,!UsesRegularization)
        nothing
    end
end

@define_traitfn UsesRegularization log_regularization!(lshaped::AbstractLShapedSolver,t::Integer) = begin
    function log_regularization!(lshaped::AbstractLShapedSolver,t::Integer,!UsesRegularization)
        nothing
    end
end

@define_traitfn UsesRegularization take_step!(lshaped::AbstractLShapedSolver) = begin
    function take_step!(lshaped::AbstractLShapedSolver,!UsesRegularization)
        nothing
    end
end

@define_traitfn UsesRegularization process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane) = begin
    function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane,!UsesRegularization)
        nothing
    end

    function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane,UsesRegularization)
        nothing
    end
end

@implement_traitfn function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane,IsRegularized)
    push!(lshaped.committee,cut)
    nothing
end

@define_traitfn UsesRegularization remove_inactive!(lshaped::AbstractLShapedSolver) = begin
    function remove_inactive!(lshaped::AbstractLShapedSolver,UsesRegularization)
        inactive = find(c->!active(lshaped,c),lshaped.committee)
        diff = length(lshaped.committee) - length(lshaped.structuredmodel.linconstr) - lshaped.nscenarios
        if isempty(inactive) || diff <= 0
            return false
        end
        if diff <= length(inactive)
            inactive = inactive[1:diff]
        end
        append!(lshaped.inactive,lshaped.committee[inactive])
        deleteat!(lshaped.committee,inactive)
        delconstrs!(lshaped.mastersolver.lqmodel,inactive)
        return true
    end
end

@define_traitfn UsesRegularization queueViolated!(lshaped::AbstractLShapedSolver) = begin
    function queueViolated!(lshaped::AbstractLShapedSolver,UsesRegularization)
        violating = find(c->violated(lshaped,c),lshaped.inactive)
        if isempty(violating)
            return false
        end
        gaps = map(c->gap(lshaped,c),lshaped.inactive[violating])
        for (c,g) in zip(lshaped.inactive[violating],gaps)
            enqueue!(lshaped.violating,c,g)
        end
        deleteat!(lshaped.inactive,violating)
        return true
    end
end

@define_traitfn UsesRegularization project!(lshaped::AbstractLShapedSolver) = begin
    function project!(lshaped::AbstractLShapedSolver,!UsesRegularization)
        nothing
    end

    function project!(lshaped::AbstractLShapedSolver,UsesRegularization)
        nothing
    end
end

# Is Regularized
# ------------------------------------------------------------
@define_traitfn IsRegularized update_objective!(lshaped::AbstractLShapedSolver)

@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,IsRegularized)
    lshaped.solverdata.σ = lshaped.parameters.σ
    push!(lshaped.σ_history,lshaped.solverdata.σ)

    update_objective!(lshaped)
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,IsRegularized)
    @unpack Q̃,σ = lshaped.solverdata
    push!(lshaped.Q̃_history,Q̃)
    push!(lshaped.σ_history,σ)
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,t::Integer,IsRegularized)
    @unpack Q̃,σ = lshaped.solverdata
    lshaped.Q̃_history[t] = Q̃
    lshaped.σ_history[t] = σ
end

@implement_traitfn function take_step!(lshaped::AbstractLShapedSolver,IsRegularized)
    @unpack Q,Q̃,θ = lshaped.solverdata
    @unpack τ,σ̅,σ̲ = lshaped.parameters
    if Q + τ*(1+abs(Q)) <= Q̃
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        if abs(Q - Q̃) <= 0.5*(Q̃-θ)
            # Enlarge the trust-region radius
            lshaped.solverdata.σ = min(σ̅,lshaped.solverdata.σ*2)
            update_objective!(lshaped)
        end
        lshaped.solverdata.major_iterations += 1
    else
        lshaped.solverdata.σ *= 0.5
        lshaped.solverdata.σ = max(σ̲,lshaped.solverdata.σ)
        update_objective!(lshaped)
        lshaped.solverdata.minor_iterations += 1
    end
    nothing
end

@implement_traitfn function update_objective!(lshaped::AbstractLShapedSolver,IsRegularized)
    # Linear regularizer penalty
    c = copy(lshaped.c)
    c -= (1/lshaped.solverdata.σ)*lshaped.ξ
    append!(c,fill(1.0,lshaped.nscenarios))
    setobj!(lshaped.mastersolver.lqmodel,c)

    # Quadratic regularizer penalty
    qidx = collect(1:length(lshaped.ξ)+lshaped.nscenarios)
    qval = fill(1/lshaped.solverdata.σ,length(lshaped.ξ))
    append!(qval,zeros(lshaped.nscenarios))
    if applicable(setquadobj!,lshaped.mastersolver.lqmodel,qidx,qidx,qval)
        setquadobj!(lshaped.mastersolver.lqmodel,qidx,qidx,qval)
    else
        error("The regularized decomposition algorithm requires a solver that handles quadratic objectives")
    end
end

# HasTrustRegion
# ------------------------------------------------------------
@define_traitfn HasTrustRegion set_trustregion!(lshaped::AbstractLShapedSolver)
@define_traitfn HasTrustRegion enlarge_trustregion!(lshaped::AbstractLShapedSolver)
@define_traitfn HasTrustRegion reduce_trustregion!(lshaped::AbstractLShapedSolver)

@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    if lshaped.parameters.autotune
        Δ = max(1.0,0.01*norm(lshaped.x,Inf))
        Δ̅ = 0.05*norm(lshaped.x,Inf)
        @pack lshaped.parameters = Δ,Δ̅
    end
    lshaped.solverdata.Δ = lshaped.parameters.Δ
    push!(lshaped.Δ_history,lshaped.solverdata.Δ)

    set_trustregion!(lshaped)
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    @unpack Q̃,Δ = lshaped.solverdata
    push!(lshaped.Q̃_history,Q̃)
    push!(lshaped.Δ_history,Δ)
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,t::Integer,HasTrustRegion)
    @unpack Q̃,Δ = lshaped.solverdata
    lshaped.Q̃_history[t] = Q̃
    lshaped.Δ_history[t] = Δ
end

@implement_traitfn function take_step!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    @unpack Q,Q̃,θ = lshaped.solverdata
    @unpack γ = lshaped.parameters
    if Q <= Q̃ - γ*abs(Q̃-θ)
        lshaped.solverdata.cΔ = 0
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        enlarge_trustregion!(lshaped)
        lshaped.solverdata.major_iterations += 1
    else
        reduce_trustregion!(lshaped)
        lshaped.solverdata.minor_iterations += 1
    end
    nothing
end

@implement_traitfn function set_trustregion!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    l = max.(lshaped.structuredmodel.colLower, lshaped.ξ-lshaped.solverdata.Δ)
    append!(l,fill(-Inf,lshaped.nscenarios))
    u = min.(lshaped.structuredmodel.colUpper, lshaped.ξ+lshaped.solverdata.Δ)
    append!(u,fill(Inf,lshaped.nscenarios))
    setvarLB!(lshaped.mastersolver.lqmodel,l)
    setvarUB!(lshaped.mastersolver.lqmodel,u)
end

@implement_traitfn function enlarge_trustregion!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    @unpack Q,Q̃,θ = lshaped.solverdata
    @unpack τ,Δ̅ = lshaped.parameters
    if abs(Q - Q̃) <= 0.5*(Q̃-θ) && norm(lshaped.ξ-lshaped.x,Inf) - lshaped.solverdata.Δ <= τ
        # Enlarge the trust-region radius
        lshaped.solverdata.Δ = min(Δ̅,2*lshaped.solverdata.Δ)
        set_trustregion!(lshaped)
        return true
    else
        return false
    end
end

@implement_traitfn function reduce_trustregion!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    @unpack Q,Q̃,θ = lshaped.solverdata
    ρ = min(1,lshaped.solverdata.Δ)*(Q-Q̃)/(Q̃-θ)
    if ρ > 0
        lshaped.solverdata.cΔ += 1
    end
    if ρ > 3 || (lshaped.solverdata.cΔ >= 3 && 1 < ρ <= 3)
        # Reduce the trust-region radius
        lshaped.solverdata.cΔ = 0
        lshaped.solverdata.Δ = (1/min(ρ,4))*lshaped.solverdata.Δ
        set_trustregion!(lshaped)
        return true
    else
        return false
    end
end

# Has Levels
# ------------------------------------------------------------

@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,HasLevels)
    # θs
    for i = 1:lshaped.nscenarios
        addvar!(lshaped.projectionsolver.lqmodel,-Inf,Inf,1.0)
    end
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,HasLevels)
    @unpack Q̃ = lshaped.solverdata
    push!(lshaped.Q̃_history,Q̃)
end

@implement_traitfn function log_regularization!(lshaped::AbstractLShapedSolver,t::Integer,HasLevels)
    @unpack Q̃ = lshaped.solverdata
    lshaped.Q̃_history[t] = Q̃
end

@implement_traitfn function take_step!(lshaped::AbstractLShapedSolver,HasLevels)
    @unpack Q,Q̃ = lshaped.solverdata
    @unpack τ = lshaped.parameters
    if Q + τ*(1+abs(Q)) <= Q̃
        lshaped.solverdata.Q̃ = Q
    end
    nothing
end

@implement_traitfn function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut},HasLevels)
    #addconstr!(lshaped.projectionsolver.lqmodel,lowlevel(cut)...) TODO: Rewrite with MathOptInterface
end

@implement_traitfn function project!(lshaped::AbstractLShapedSolver,HasLevels)
    @unpack θ,Q̃ = lshaped.solverdata
    @unpack λ = lshaped.parameters
    # Copy current master problem (TODO: Rewrite with MathOptInterface)
    lshaped.projectionsolver.lqmodel = copy(lshaped.mastersolver.lqmodel)
    # Update level
    c = sparse(getobj(lshaped.projectionsolver.lqmodel))
    L = (1-λ)*Q̃ + λ*θ
    addconstr!(lshaped.projectionsolver.lqmodel,c.nzind,c.nzval,-Inf,L)
e
    # Update regularizer
    q = -copy(lshaped.ξ)
    append!(q,zeros(lshaped.nscenarios))
    setobj!(lshaped.projectionsolver.lqmodel,q)
    # Quadratic regularizer penalty
    qidx = collect(1:length(lshaped.ξ)+lshaped.nscenarios)
    qval = ones(length(lshaped.ξ))
    append!(qval,zeros(lshaped.nscenarios))
    if applicable(setquadobj!,lshaped.projectionsolver.lqmodel,qidx,qidx,qval)
        setquadobj!(lshaped.projectionsolver.lqmodel,qidx,qidx,qval)
    else
        error("The level set algorithm requires a solver that handles quadratic objectives")
    end
    # Solve projection problem
    lshaped.projectionsolver(lshaped.x)
    if status(lshaped.projectionsolver) == :Infeasible
        error("Projection problem is infeasible, aborting procedure.")
    end
    # Update master solution
    ncols = lshaped.structuredmodel.numCols
    x = getsolution(lshaped.projectionsolver)
    lshaped.x[1:ncols] = x[1:ncols]
    lshaped.ξ[:] = lshaped.x[:]
end
