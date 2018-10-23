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

@define_traitfn UsesRegularization decision(lshaped::AbstractLShapedSolver) = begin
    function decision(lshaped::AbstractLShapedSolver,!UsesRegularization)
        return lshaped.x
    end

    function decision(lshaped::AbstractLShapedSolver,UsesRegularization)
        return lshaped.ξ
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
            @pack! lshaped.parameters = σ,σ̅,σ̲
        else
            σ̅ = max(4.0,norm(lshaped.x))
            σ̲ = min(2.0,0.005*norm(lshaped.x))
            σ = σ̲
            @pack! lshaped.parameters = σ,σ̅,σ̲
        end
    end
    lshaped.solverdata.σ = lshaped.parameters.σ
    push!(lshaped.σ_history,lshaped.solverdata.σ)
    # Add ∞-norm auxilliary variable
    if lshaped.parameters.linearize
        # t
        MPB.addvar!(lshaped.mastersolver.lqmodel,-Inf,Inf,1.0)
        if typeof(lshaped.mastersolver.optimsolver) == GurobiSolver
            updatemodel!(lshaped.mastersolver.lqmodel)
        end
    end
    # Add quadratic penalty
    c = copy(lshaped.c)
    append!(c,fill(1.0,nbundles(lshaped)))
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
    @unpack τ,γ,σ̅,σ̲ = lshaped.parameters
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
        append!(c,fill(1.0,nbundles(lshaped)))
        add_penalty!(lshaped,lshaped.mastersolver.lqmodel,c,1/lshaped.solverdata.σ,lshaped.ξ)
    end
    nothing
end

@implement_traitfn function solve_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver,IsRegularized)
    if lshaped.parameters.linearize
        solve_linearized_problem!(lshaped,solver)
    else
        solver(lshaped.mastervector)
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
        Δ̅ = max(1000.0,norm(lshaped.x,Inf))
        @pack! lshaped.parameters = Δ,Δ̅
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
    nb = nbundles(lshaped)
    l = max.(lshaped.structuredmodel.colLower, lshaped.ξ .- lshaped.solverdata.Δ)
    append!(l,fill(-Inf,nb))
    u = min.(lshaped.structuredmodel.colUpper, lshaped.ξ .+ lshaped.solverdata.Δ)
    append!(u,fill(Inf,nb))
    MPB.setvarLB!(lshaped.mastersolver.lqmodel,l)
    MPB.setvarUB!(lshaped.mastersolver.lqmodel,u)
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

@implement_traitfn function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane{FeasibilityCut},HasTrustRegion)
    @unpack τ = lshaped.parameters
    if !satisfied(cut,lshaped.ξ,τ)
        A = [I cut.δQ; cut.δQ' 0*I]
        b = [zeros(length(lshaped.ξ)); -gap(cut,lshaped.ξ)]
        t = A\b
        lshaped.ξ[:] = lshaped.ξ + t[1:length(lshaped.ξ)]
    end
end

# Has Levels
# ------------------------------------------------------------

@implement_traitfn function init_solver!(lshaped::AbstractLShapedSolver,HasLevels)
    # θs
    for i = 1:nbundles(lshaped)
        MPB.addvar!(lshaped.projectionsolver.lqmodel,-Inf,Inf,0.0)
    end
    if lshaped.parameters.linearize
        # t
        MPB.addvar!(lshaped.projectionsolver.lqmodel,-Inf,Inf,1.0)
    end
    if typeof(lshaped.projectionsolver.optimsolver) == GurobiSolver
        updatemodel!(lshaped.projectionsolver.lqmodel)
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
    if lshaped.parameters.linearize
        solve_linearized_problem!(lshaped,solver)
    else
        solver(lshaped.mastervector)
    end
end

@implement_traitfn function process_cut!(lshaped::AbstractLShapedSolver,cut::HyperPlane,HasLevels)
    MPB.addconstr!(lshaped.projectionsolver.lqmodel,lowlevel(cut)...)
    # TODO: Rewrite with MathOptInterface
end

@implement_traitfn function project!(lshaped::AbstractLShapedSolver,HasLevels)
    @unpack θ,Q̃ = lshaped.solverdata
    @unpack λ = lshaped.parameters
    # Update level (TODO: Rewrite with MathOptInterface)
    nb = nbundles(lshaped)
    c = sparse(MPB.getobj(lshaped.mastersolver.lqmodel))
    L = (1-λ)*θ + λ*Q̃
    push!(lshaped.levels,L)
    if lshaped.solverdata.levelindex == -1
        MPB.addconstr!(lshaped.projectionsolver.lqmodel,c.nzind,c.nzval,-Inf,L)
        lshaped.solverdata.levelindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
    else
        MPB.delconstrs!(lshaped.projectionsolver.lqmodel,[lshaped.solverdata.levelindex])
        MPB.addconstr!(lshaped.projectionsolver.lqmodel,c.nzind,c.nzval,-Inf,L)
        lshaped.solverdata.levelindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
        if lshaped.parameters.linearize
            if typeof(lshaped.projectionsolver.optimsolver) == GurobiSolver
                updatemodel!(lshaped.projectionsolver.lqmodel)
            end
            lshaped.solverdata.regularizerindex -= 1
        end
    end
    # Update regularizer
    add_penalty!(lshaped,lshaped.projectionsolver.lqmodel,zeros(length(lshaped.ξ)+nb),1.0,lshaped.ξ)
    # Solve projection problem
    solve_problem!(lshaped,lshaped.projectionsolver)
    if status(lshaped.projectionsolver) == :Infeasible
        @warn "Projection problem is infeasible, unprojected solution will be used"
        if Q̃ <= θ
            # If the upper objective bound is lower than the model lower bound for some reason, reset it.
            lshaped.solverdata.Q̃ = Inf
        end
    else
        # Update master solution
        ncols = lshaped.structuredmodel.numCols
        x = getsolution(lshaped.projectionsolver)
        lshaped.mastervector[:] = x[1:ncols+nb]
        lshaped.x[1:ncols] = x[1:ncols]
        lshaped.θs[:] = x[ncols+1:ncols+nb]
    end
    nothing
end

function add_penalty!(lshaped::AbstractLShapedSolver,model::MPB.AbstractLinearQuadraticModel,c::AbstractVector,α::Real,ξ::AbstractVector)
    nb = nbundles(lshaped)
    if lshaped.parameters.linearize
        ncols = lshaped.structuredmodel.numCols
        tidx = ncols+nb+1
        j = lshaped.solverdata.regularizerindex
        if j == -1
            for i in 1:ncols
                MPB.addconstr!(model,[i,tidx],[-α,1],-α*ξ[i],Inf)
                MPB.addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
            end
        else
            MPB.delconstrs!(model,collect(j:j+2*ncols-1))
            for i in 1:ncols
                MPB.addconstr!(model,[i,tidx],[-α,1],-ξ[i],Inf)
                MPB.addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
            end
        end
        lshaped.solverdata.regularizerindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
        if hastrait(lshaped,HasLevels)
            lshaped.solverdata.regularizerindex += 1
        end
    else
        # Linear part
        c[1:length(ξ)] -= α*ξ
        MPB.setobj!(model,c)
        # Quadratic part
        qidx = collect(1:length(ξ)+nb)
        qval = fill(α,length(lshaped.ξ))
        append!(qval,zeros(nb))
        if applicable(MPB.setquadobj!,model,qidx,qidx,qval)
            MPB.setquadobj!(model,qidx,qidx,qval)
        else
            error("Setting a quadratic penalty requires a solver that handles quadratic objectives")
        end
    end
end

function solve_linearized_problem!(lshaped::AbstractLShapedSolver,solver::LQSolver)
    push!(lshaped.mastervector,norm(lshaped.x-lshaped.ξ,Inf))
    solver(lshaped.mastervector)
    pop!(lshaped.mastervector)
    return nothing
end
