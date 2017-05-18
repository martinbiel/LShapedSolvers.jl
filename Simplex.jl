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
#            println("Optimal solution found at iteration ",iter,"!")
 #           @printf("Optimal value = %12.5e \n",obj)
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
  #          println("Problem is unbounded, aborting procedure at iteration ",iter)
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
   #     println( "\n  Iter     Objective  Leaving Entering           Step \n" )
    #    @printf( " %5g   %12.5e", iter, obj );
     #   @printf( " %8g   %6g   %12.2e\n", rout, qin, maxstep );

        # Break because of possible cycling
        if (iter > exp(n))
      #      println("The algorithm probably cycles, aborting procedure\n")
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
