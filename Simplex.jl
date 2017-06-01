function primalsimplex(A::SparseMatrixCSC,
                       b::AbstractVector,
                       c::AbstractVector,
                       basis::Vector{Int})
    n = size(A,2)
    m = size(A,1)
    @assert(m == length(b), "Dimension error in b")
    @assert(n == length(c), "Dimension error in c")
    @assert(m == length(basis), "Incorrect set of basis variables do not form a basis")
    iter = 0
    status = :NotSolved
    obj = 0
    x = zeros(n)
    y = zeros(m)
    s = zeros(n)
    nonbasis = collect(1:n)
    sort!(basis)
    deleteat!(nonbasis,basis)

    # ================
    # EXPAND procedure variables
    df = eps()^(3/8)
    K = eps()^(-1/4)
    d0 = 0.5*df
    dk = d0
    dK = 0.99*df
    tau = (dK-d0)/K
    tol = eps()^(2/3)
    expanditer = 0

    # EXPAND procedure functions
    step(x,p,l,tol) = begin
       s = Inf
       if (p < -tol && l > -Inf)
           s = (l-x)/p
       end
       return s
    end

    ratioTest(x,p,l,tol) = begin
        j = 0
        alpha = Inf;
        for rati = eachindex(x)
           alphaj = step(x[rati],p[rati],l,tol)
           if (alphaj < alpha)
               alpha = alphaj
               j = rati
           end
        end
        return alpha,j
    end
    # ================

    # Simplex procedure
    # =================
    while(true)
        iter = iter + 1

        # Update matrices
        B = A[:,basis]
        N = A[:,nonbasis]

        # Calculate first iterate
        if (iter == 1)
            xb = B\b
            x[basis] = xb
            @assert(all(xb .> -d0),"Initial basis must be primal feasible")
        end

        # Calculate current objective
        obj = c⋅x

        # Calculate reduced cost
        s = zeros(n)
        y = B'\c[basis]
        sn = c[nonbasis]-N'*y
        s[nonbasis] = sn

        if (all(sn .>= -(d0+tau)) && all(x[basis] .> (-d0-tau)))
            if (any(x[nonbasis] .< (-d0-tau)))
                # Could be optimal, reset EXPAND procedure and check again
                count = 0
                for resi = nonbasis
                   if (x[resi] < -eps()^(2/3))
                       count = count + 1
                   end
                   x[resi] = 0
                end
                if (count > 0)
                    xb = B\b
                    x[basis] = xb
                end
                # Do not count this as an iteration
                iter = iter - 1
                continue;
            end
            # Optimal!
            println("Optimal solution found at iteration ",iter,"!")
            @printf("Optimal value = %12.5e \n",obj)
            status = :Optimal
            return x,obj,y,s,basis,iter,status
        end

        # Largest reduced cost
        i = indmin(sn)
        qin = nonbasis[i]
        # -> x_q should enter the basis

        # Calculate search direction
        pn = zeros(length(nonbasis))
        pn[i] = 1
        pb = -B\(N*pn)
        p = zeros(n)
        p[basis] = pb
        p[nonbasis] = pn

        if(all(pb .> -(d0+tau)))
            # Unbounded!
            obj = -Inf
            x = []
            println("Problem is unbounded, aborting procedure at iteration ",iter)
            status = :Unbounded
            break
        end

        # EXPAND procedure
        # ================
        expanditer = expanditer + 1
        if (expanditer == K)
            # Reset EXPAND procedure
            count = 0
            for resi = nonbasis
               if (x[resi] < -eps()^(2/3))
                   count = count + 1
               end
               x[resi] = 0
            end
            if (count > 0)
                xb = B\b
                x[basis] = xb
            end
            expanditer = 0
            dk = d0
        end
        # Update tolerance
        dk = dk + tau

        # EXPAND
        alpha1,_ = ratioTest(x,p,-dk,tol)
        alpha2 = 0
        r = 0
        pmax = 0
        for expi = eachindex(basis)
            pl = p[basis[expi]]
            alphal = step(x[basis[expi]],pl,0,tol)
            if (alphal <= alpha1 && abs(pl) > pmax)
                r = expi
                alpha2 = alphal
                pmax = abs(pl)
            end
        end
        alphamin = tau/pmax
        maxstep = max(alpha2, alphamin)
        # ================

        # -> x_r leaves the basis
        rout = basis[r]

        # Update iterate
        x = x + maxstep*p

        # Update basis
        nonbasis[i] = rout
        basis[r] = qin

        # Print iteration information
            println( "\n  Iter     Objective  Leaving Entering           Step \n" )
            @printf( " %5g   %12.5e", iter, obj );
            @printf( " %8g   %6g   %12.2e\n", rout, qin, maxstep );

        # Break because of possible cycling
        if (iter > exp(n))
            println("The algorithm probably cycles, aborting procedure\n")
            status = :NotSolved
            break
        end
    end
    return x,obj,y,s,basis,iter,status
end

function primalsimplex(A::SparseMatrixCSC,
    b::AbstractVector,
    c::AbstractVector)
    m,n = size(A);
    fuzz = sqrt(eps());
    iter1 = 0
    iter2 = 0
    status = :NotSolved

    A1 = [ A diagm(sign(b+fuzz)) ];
    c1 = [ zeros(n) ; ones(m) ];
    basis = collect((n+1):(n+m));

    xfeas,suminf,y,s,basis,iter1,_ = primalsimplex(A1,b,c1,basis)

    if suminf > fuzz
        println("Error, no feasible solution exists \n")
        x = []
        y = []
        s = []
        obj = Inf
        status = :Infeasible
    else
        if maximum(basis) > n
            println("Warning, artificial variables are still basic")
            println("Cannot be dealt with properly in this implementation \n")
            basart = basis[find(basis>n)]
            basis  = [ basis[find(basis<=n)] (n+1):(n+length(basart)) ]
            x,obj,y,s,basis,iter2,status = primalsimplex(A1[:,[1:n basart]],b,[c;1e6*ones(size(basart'))],basis)
            x = x[1:n]
            s = s[1:n]
        else
            x,obj,y,s,basis,iter2,status = primalsimplex(A,b,c,basis)
        end
    end
    return x,obj,y,s,basis,iter1,iter2,status
end


typealias ShortStep Val{:ShortStep}
typealias LongStep Val{:LongStep}
typealias MethodType Union{
    Type{ShortStep},
    Type{LongStep}}

immutable IPparams
    ζ::Float64
    ϵ::Float64
    τ::Float64
    γ::Float64
    method::MethodType
end
IP_default_params() = IPparams(20,1e-8,1e-5,1e6,LongStep)

function interiorpoint{T <: Real}(A::AbstractMatrix{T},
                                  b::AbstractVector{T},
                                  c::AbstractVector{T},
                                  params::IPparams = IP_default_params())
    # Assert proper form of LP
    m = size(A,1)
    n = size(A,2)
    status = :NotSolved
    @assert(m == length(b), "Dimension error in b")
    @assert(n == length(c), "Dimension error in c")
    # Pick initial starting point
    x = params.ζ*ones(n)
    λ = zeros(m)
    s = params.ζ*ones(n)
    obj = c⋅x
    μ = (x⋅s)/n
    r_c = A'*λ+s-c
    r_b = A*x-b
    iter = 0

    x_iterates = []
    s_iterates = []

    # Pick σ
    σ = 0
    if (params.method == ShortStep)
        σ = 1-0.4/sqrt(length(x))
    elseif (params.method == LongStep)
        σ = 0.1
    end

    # Main loop
    while(true)
        iter += 1

        # Terminate?
        if (norm(r_b)/(1+norm(b)) <= params.ϵ &&
            norm(r_c)/(1+norm(c)) <= params.ϵ &&
            abs(c⋅x-b⋅λ)/(1+abs(c⋅x)) <= params.ϵ)
            # Optimal!
            status = :Optimal
            println("Optimal solution found at iteration ",iter,"!")
            print("xopt = \n")
            println(reshape(x,length(x),1))
            @printf("Optimal value = %12.5e",obj)
            break
        end

        # Construct new normal system
        S = diagm(s)
        X = diagm(x)
        AA = [zeros(n,n) A' eye(n);
             A zeros(m,m) zeros(m,n);
             S zeros(n,m) X]
        g = [-r_c; -r_b; -X*S*ones(n) + σ*μ*ones(n)]

        if(det(AA) ≈ 0)
            println("Premature termination since system matrix is singular.")
            break
        end

        # Solve for new direction
        p = AA\g
        Δx = p[1:n]
        Δλ = p[n+1:m+n]
        Δs = p[m+n+1:end]

        # Termination?
        if (norm(Δx) <= params.τ && norm(Δs) <= params.τ)
            # Optimal!
            println("Optimal solution found at iteration ",iter,"!")
            print("xopt = \n")
            println(reshape(x,length(x),1))
            @printf("Optimal value = %12.5e",obj)
            break
        end

        Δx_inds = Δx .< -params.τ
        Δs_inds = Δs .< -params.τ
        if (!any(Δx_inds) && !any(Δs_inds))
            # Probably round off error
            println("Premature termination since search direction is positive or very small, probably due to round off errors or infeasibility.")
            break
        end
        # Step length from simple linesearch
        α = min(1,0.99*min((-x./Δx)[Δx_inds]...,(-s./Δs)[Δs_inds]...))
        if (α <= params.ϵ)
            # Probably converged
            println("Premature termination since steplength is below tolerance, the algorithm may have converged or stalled.")
            break
        end
        # Update the iterates
        x = x + α*Δx
        λ = λ + α*Δλ
        s = s + α*Δs
        obj = c⋅x
        μ = (x⋅s)/n
        r_c = A'*λ+s-c
        r_b = A*x-b

        # Print iteration information
        println( "\n  Iter     Objective     Step " )
        @printf( " %5g   %12.5e %12.2e\n", iter, obj, α );

        # Assume infeasible if x or s seems to diverge
        if (norm(x) >= params.γ || norm(s) >= params.γ)
            x = []
            status = :Infeasible
            println("Problem is infeasible, aborting procedure at iteration ",iter)
            break
        end

        push!(x_iterates,norm(x))
        push!(s_iterates,norm(s))
    end
    return x,obj,λ,s,iter,x_iterates,s_iterates,status
end
