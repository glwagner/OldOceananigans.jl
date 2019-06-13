#
# Constant diffusivity
#

"""
    ConstantIsotropicDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

or

    MolecularDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

Return a `ConstantIsotropicDiffusivity` closure object of type `T` with
viscosity `ν` and scalar diffusivity `κ`.
"""
Base.@kwdef struct ConstantIsotropicDiffusivity{T} <: IsotropicDiffusivity{T}
    ν :: T = 1e-6
    κ :: T = 1e-7
end

const MolecularDiffusivity = ConstantIsotropicDiffusivity

ConstantIsotropicDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, ConstantIsotropicDiffusivity; kwargs...)

calc_diffusivities!(diffusivities, grid, closure::ConstantIsotropicDiffusivity,
                         args...) = nothing

# These functions are used to specify Gradient and Value boundary conditions.
@inline κ_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.κ
@inline ν_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν

@inline ∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::ConstantIsotropicDiffusivity, args...) = (
      closure.κ / grid.Δx^2 * δx²_c2f2c(grid, ϕ, i, j, k)
    + closure.κ / grid.Δy^2 * δy²_c2f2c(grid, ϕ, i, j, k)
    + closure.κ / grid.Δz^2 * δz²_c2f2c(grid, ϕ, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2c2f(grid, u, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2e2f(grid, u, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2e2f(grid, u, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2e2f(grid, v, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2c2f(grid, v, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2e2f(grid, v, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2e2f(grid, w, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2e2f(grid, w, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2c2f(grid, w, i, j, k)
)
