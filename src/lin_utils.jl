
using LinearAlgebra, ForwardDiff

## Make Jacobian and residual fns of f via ForwardDiff.jl
function linearize(f, nx, nu, nd; solve_lg=true)

    g(x) = f(zero(x), x, 0.0, 0.0)
    A(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, 1:nx]
    B₁(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, nx+1:nx+nu]
    B₂(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, nx+nu+1:nx+nu+nd]
    # c(x̃) = g(x̃)[1:nx] - A(x̃) * x̃[1:nx] - B₁(x̃) * x̃[nx+1:nx+nu] - B₂(x̃) * x̃[nx+nu+1:nx+nu+nd]
    c(x̃) = (g(x̃) - ForwardDiff.jacobian(g,x̃) * x̃)[1:nx]
    
    linear_graph = falses(nx, nx+nu+nd)
    if solve_lg
        for i=1:nx, j=1:nx+nu+nd; linear_graph[i,j] = ForwardDiff.jacobian(g, Inf*((1:nx+nu+nd).==j))[i,:] == ForwardDiff.jacobian(g,zeros(nx+nu+nd))[i,:]; end
    end

    return (A, B₁, B₂, c), linear_graph
end

## Make Hessian fns of f via ForwardDiff.jl
function hessians(f; solve_dims=solve_dims, affine_inputs=false)
    Gs = []
    for i in solve_dims
        gi(x) = f(zero(x), x, 0.0, 0.0)[i]
        Gi(ξ) = ForwardDiff.hessian(gi, ξ)
        push!(Gs, Gi)
    end
    return Gs
end