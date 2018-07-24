# ------------------------------------------------------------
# UsesRegularization: Algorithm uses some regularization method
# ------------------------------------------------------------
@define_trait UsesRegularization = begin
    IsRegularized  # Algorithm uses the regularized decomposition method of Ruszczyński
    HasTrustRegion # Algorithm uses the trust-region method of Linderoth/Wright
    HasLevels      # Algorithm uses the level set method of Fábián/Szőke
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

@define_traitfn UsesRegularization solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver) = begin
    function solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver,!UsesRegularization)
        solver(lshaped.mastervector)
    end

    function solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver,UsesRegularization)
        solver(lshaped.mastervector)
    end
end

@define_traitfn UsesRegularization gap(lshaped::AbstractLShapedSolver) = begin
    function gap(lshaped::AbstractLShapedSolver,!UsesRegularization)
        @unpack τ = lshaped.parameters
        @unpack Q,θ = lshaped.solverdata
        return abs(θ-Q)/(abs(Q)+1e-10)
    end

    function gap(lshaped::AbstractLShapedSolver,UsesRegularization)
        @unpack τ = lshaped.parameters
        @unpack Q̃,θ = lshaped.solverdata
        return abs(θ-Q̃)/(abs(Q̃)+1e-10)
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
@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,IsRegularized)
    if lshaped.parameters.autotune
        if lshaped.parameters.linearize
            σ̅ = max(4.0,0.01*norm(lshaped.x,Inf))
            σ̲ = min(2.0,0.001*norm(lshaped.x,Inf))
            σ = min(3.0,0.005*norm(lshaped.x,Inf))
            @pack lshaped.parameters = σ,σ̅,σ̲
        else
            σ̅ = max(4.0,norm(lshaped.x))
            σ̲ = min(2.0,0.005*norm(lshaped.x))
            σ = σ̲
            @pack lshaped.parameters = σ,σ̅,σ̲
        end
    end
    lshaped.solverdata.σ = lshaped.parameters.σ
    push!(lshaped.σ_history,lshaped.solverdata.σ)
    # Add ∞-norm auxilliary variable
    if lshaped.parameters.linearize
        # t
        addvar!(lshaped.mastersolver.lqmodel,-Inf,Inf,1.0)
    end
    # Add quadratic penalty
    c = copy(lshaped.c)
    append!(c,fill(1.0,lshaped.nscenarios))
    add_penalty!(lshaped,lshaped.mastersolver.lqmodel,c,1/lshaped.solverdata.σ,lshaped.ξ)
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
    @unpack Q,Q̃,θ,σ = lshaped.solverdata
    @unpack τ,σ̅,σ̲ = lshaped.parameters
    need_update = false
    if abs(θ-Q) <= τ*(1+abs(θ)) || lshaped.solverdata.major_iterations == 0
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        need_update = true
        lshaped.solverdata.major_iterations += 1
    else
        lshaped.solverdata.minor_iterations += 1
    end
    new_σ = if Q + τ <= (1-γ)*Q̃ + γ*θ
        min(σ̅,2*σ)
    elseif Q - τ >= γ*Q̃ + (1-γ)*θ
        max(σ̲,0.5*σ)
    else
        σ
    end
    if abs(new_σ-σ) > τ
        need_update = true
    end
    lshaped.solverdata.σ = new_σ
    if need_update
        c = copy(lshaped.c)
        append!(c,fill(1.0,lshaped.nscenarios))
        add_penalty!(lshaped,lshaped.mastersolver.lqmodel,c,1/lshaped.solverdata.σ,lshaped.ξ)
    end
    nothing
end

@implement_traitfn function solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver,IsRegularized)
    solve_linearized_problem!(lshaped,solver)
end

# HasTrustRegion
# ------------------------------------------------------------
@define_traitfn HasTrustRegion set_trustregion!(lshaped::AbstractLShapedSolver)
@define_traitfn HasTrustRegion enlarge_trustregion!(lshaped::AbstractLShapedSolver)
@define_traitfn HasTrustRegion reduce_trustregion!(lshaped::AbstractLShapedSolver)

@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,HasTrustRegion)
    if lshaped.parameters.autotune
        Δ = max(1.0,0.01*norm(lshaped.x,Inf))
        Δ̅ = max(1000.0,norm(lshaped.x,Inf))
        @pack lshaped.parameters = Δ,Δ̅
    end
    lshaped.solverdata.Δ = lshaped.parameters.Δ
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
    need_update = false
    if Q <= Q̃ - γ*abs(Q̃-θ)
        need_update = true
        enlarge_trustregion!(lshaped)
        lshaped.solverdata.cΔ = 0
        lshaped.ξ[:] = lshaped.x[:]
        lshaped.solverdata.Q̃ = Q
        lshaped.solverdata.major_iterations += 1
    else
        need_update = reduce_trustregion!(lshaped)
        lshaped.solverdata.minor_iterations += 1
    end
    if need_update
        set_trustregion!(lshaped)
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
    if Q̃ - Q >= 0.5*(Q̃-θ) && abs(norm(lshaped.ξ-lshaped.x,Inf) - lshaped.solverdata.Δ) <= τ
        # Enlarge the trust-region radius
        lshaped.solverdata.Δ = min(Δ̅,2*lshaped.solverdata.Δ)
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
        addvar!(lshaped.projectionsolver.lqmodel,-Inf,Inf,0.0)
    end
    if lshaped.parameters.linearize
        # t
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
    if Q + τ <= Q̃
        lshaped.solverdata.Q̃ = Q
        lshaped.ξ[:] = lshaped.x[:]
    end
    nothing
end

@implement_traitfn function solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver,HasLevels)
    solve_linearized_problem!(lshaped,solver)
end

@implement_traitfn function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{OptimalityCut},HasLevels)
    addconstr!(lshaped.projectionsolver.lqmodel,lowlevel(cut)...)
    # TODO: Rewrite with MathOptInterface
end

@implement_traitfn function project!(lshaped::AbstractLShapedSolver,HasLevels)
    @unpack θ,Q̃ = lshaped.solverdata
    @unpack λ = lshaped.parameters
    # Update level (TODO: Rewrite with MathOptInterface)
    c = sparse(getobj(lshaped.mastersolver.lqmodel))
    L = (1-λ)*θ + λ*Q̃
    push!(lshaped.levels,L)
    if lshaped.solverdata.levelindex == -1
        addconstr!(lshaped.projectionsolver.lqmodel,c.nzind,c.nzval,-Inf,L)
        lshaped.solverdata.levelindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
    else
        delconstrs!(lshaped.projectionsolver.lqmodel,lshaped.solverdata.levelindex)
        addconstr!(lshaped.projectionsolver.lqmodel,c.nzind,c.nzval,-Inf,L)
        lshaped.solverdata.levelindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
    end
    # Update regularizer
    add_penalty!(lshaped,lshaped.projectionsolver.lqmodel,zeros(length(lshaped.ξ)+lshaped.nscenarios),1.0,lshaped.ξ)
    # Solve projection problem
    solve_problem!(lshaped,lshaped.projectionsolver)
    if status(lshaped.projectionsolver) == :Infeasible
        error("Projection problem is infeasible, aborting procedure.")
    end
    # Update master solution
    ncols = lshaped.structuredmodel.numCols
    x = getsolution(lshaped.projectionsolver)
    lshaped.mastervector[:] = x[1:ncols+lshaped.nscenarios]
    lshaped.x[1:ncols] = x[1:ncols]
    lshaped.θs[:] = x[ncols+1:ncols+lshaped.nscenarios]
    nothing
end

function add_penalty!(lshaped::AbstractLShapedSolver,model::AbstractLinearQuadraticModel,c::AbstractVector,α::Real,ξ::AbstractVector)
    if lshaped.parameters.linearize
        ncols = lshaped.structuredmodel.numCols
        tidx = ncols+nscenarios(lshaped)+1
        j = lshaped.solverdata.regularizerindex
        if j == -1
            for i in 1:ncols
                addconstr!(model,[i,tidx],[-α,1],-α*ξ[i],Inf)
                addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
            end
        else
            for i in j:j+2*ncols-1
                delconstrs!(model,i)
            end
            for i in 1:ncols
                addconstr!(model,[i,tidx],[-α,1],-ξ[i],Inf)
                addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
            end
        end
        lshaped.solverdata.regularizerindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
        if hastrait(lshaped,HasLevels)
            lshaped.solverdata.regularizerindex += 1
        end
    else
        # Linear part
        c[1:length(ξ)] -= α*ξ
        setobj!(model,c)
        # Quadratic part
        qidx = collect(1:length(ξ)+lshaped.nscenarios)
        qval = fill(α,length(lshaped.ξ))
        append!(qval,zeros(lshaped.nscenarios))
        if applicable(setquadobj!,model,qidx,qidx,qval)
            setquadobj!(model,qidx,qidx,qval)
        else
            error("Setting a quadratic penalty requires a solver that handles quadratic objectives")
        end
    end
end

function solve_linearized_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver)
    push!(lshaped.mastervector,norm(lshaped.x-lshaped.ξ,Inf))
    solver(lshaped.mastervector)
    pop!(lshaped.mastervector)
end
