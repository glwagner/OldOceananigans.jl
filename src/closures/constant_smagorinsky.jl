Base.@kwdef struct ConstantSmagorinsky{T} <: IsotropicDiffusivity{T}
              Cs :: T = 0.2
              Cb :: T = 1.0
              Pr :: T = 1.0
    ν_background :: T = 1e-6
    κ_background :: T = 1e-7
end

"""
    ConstantSmagorinsky(T=Float64; C=0.23, Pr=1.0, ν_background=1e-6,
                            κ_background=1e-7)

Returns a `ConstantSmagorinsky` closure object of type `T` with

    * `C`            : Smagorinsky constant
    * `Pr`           : Prandtl number
    * `ν_background` : background viscosity
    * `κ_background` : background diffusivity

"""
ConstantSmagorinsky(T; kwargs...) =
      typed_keyword_constructor(T, ConstantSmagorinsky; kwargs...)

"Return the filter width for Constant Smagorinsky on a Regular Cartesian grid."
@inline Δ(i, j, k, grid::RegularCartesianGrid{T}, ::ConstantSmagorinsky{T}) where T = geo_mean_Δ(grid)

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::ConstantSmagorinsky)
    ν_ccc = zeros(arch, grid)
    return (ν_ccc=ν_ccc, )
end

"Return the double dot product of strain at `ccc`."
@inline function ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_cca(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xz_cac(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶yz_acc(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
const Δ_ccc = Δ
const Δ_ffc = Δ
const Δ_fcf = Δ
const Δ_cff = Δ

"""
    buoyancy(i, j, k, grid, eos, g, T, S)

Calculate the buoyancy at grid point `i, j, k` associated with `eos`,
gravitational acceleration `g`, temperature `T`,  and salinity `S`.
"""
@inline buoyancy(i, j, k, grid, eos::LinearEquationOfState, grav, T, S) =
    grav * ( eos.βT * (T[i, j, k] - eos.T₀) - eos.βS * (S[i, j, k] - eos.S₀) )

"""
    stability(N², Σ², Pr, Cb)

Return the stability function

``1 - Cb N^2 / (Pr Σ^2)``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::T, Σ²::T, Pr::T, Cb::T) where T = one(T) - sqrt(stability_factor(N², Σ², Pr, Cb))
@inline stability_factor(N²::T, Σ²::T, Pr::T, Cb::T) where T = min(one(T), max(zero(T), Cb * N² / (Pr*Σ²)))

"""
    νₑ(ς, Cs, Δ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `Cs`,
filter with `Δ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ(ς, Cs, Δ, Σ²) = ς * (Cs*Δ)^2 * sqrt(2Σ²)

@inline function ν_ccc(i, j, k, grid, clo::ConstantSmagorinsky, ϕ, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, grav, T, S)
     Δ = Δ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν_background
end

@inline function κ_ccc(i, j, k, grid, clo::ConstantSmagorinsky, ϕ, eos, grav, u, v, w, T, S)
    νₑ = ν_ccc(i, j, k, grid, clo, ϕ, eos, grav, u, v, w, T, S)
    return (νₑ - clo.ν_background) / clo.Pr + clo.κ_background
end

"""
    κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂x ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_ϕ(i, j, k, grid, ϕ, ν, closure::ConstantSmagorinsky)
    ν = ▶x_faa(i, j, k, grid, ν, closure)
    κ = (ν - closure.ν_background) / closure.Pr + closure.κ_background
    ∂x_ϕ = ∂x_faa(i, j, k, grid, ϕ)
    return κ * ∂x_ϕ
end

"""
    κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure::ConstantSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂y ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_ϕ(i, j, k, grid, ϕ, ν, closure::ConstantSmagorinsky)
    ν = ▶y_afa(i, j, k, grid, ν, closure)
    κ = (ν - closure.ν_background) / closure.Pr + closure.κ_background
    ∂y_ϕ = ∂y_afa(i, j, k, grid, ϕ)
    return κ * ∂y_ϕ
end

"""
    κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure::ConstantSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂z ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_ϕ(i, j, k, grid, ϕ, ν, closure::ConstantSmagorinsky)
    ν = ▶z_aaf(i, j, k, grid, ν, closure::ConstantSmagorinsky)
    κ = (ν - closure.ν_background) / closure.Pr + closure.κ_background
    ∂z_ϕ = ∂z_aaf(i, j, k, grid, ϕ)
    return κ * ∂z_ϕ
end

"""
    ∇_κ_∇_ϕ(i, j, k, grid, ϕ, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ ϕ)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::ConstantSmagorinsky, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_ϕ, ϕ, diffusivities.ν_ccc, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_ϕ, ϕ, diffusivities.ν_ccc, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_ϕ, ϕ, diffusivities.ν_ccc, closure)
)

#=
"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantSmagorinsky, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₁₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, u, v, w, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantSmagorinsky, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₂₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, u, v, w, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, diffusivities)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantSmagorinsky, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₃₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, u, v, w, diffusivities)
)
=#

function calculate_diffusivities!(diffusivities, grid, closure::ConstantSmagorinsky, eos, grav, u, v, w, T, S)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds diffusivities.ν_ccc[i, j, k] = ν_ccc(i, j, k, grid, closure, eos, grav, u, v, w, T, S)
            end
        end
    end
    @synchronize
end
