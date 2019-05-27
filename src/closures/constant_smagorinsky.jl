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
Δ(i, j, k, grid::RegularCartesianGrid, ::ConstantSmagorinsky) = geo_mean_Δ(grid)

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

"Return the double dot product of strain at `ccc`."
function ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_cca(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xz_cac(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶yz_acc(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
function ΣᵢⱼΣᵢⱼ_ffc(i, j, k, grid, u, v, w)
    return (
                  ▶xy_ffa(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *    Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ▶yz_afc(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶xz_fac(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
function ΣᵢⱼΣᵢⱼ_fcf(i, j, k, grid, u, v, w)
    return (
                  ▶xz_faf(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶yz_acf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *    Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_fca(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
function ΣᵢⱼΣᵢⱼ_cff(i, j, k, grid, u, v, w)
    return (
                  ▶yz_aff(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶xz_caf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xy_cfa(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *    Σ₂₃²(i, j, k, grid, u, v, w)
            )
end

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
Δ_ccc = Δ
Δ_ffc = Δ
Δ_fcf = Δ
Δ_cff = Δ

"""
    buoyancy(i, j, k, grid, eos, g, T, S)

Calculate the buoyancy at grid point `i, j, k` associated with `eos`, 
gravitational acceleration `g`, temperature `T`,  and salinity `S`.
"""
buoyancy(i, j, k, grid, eos::LinearEquationOfState, g, T, S) = 
    g * ( eos.βT * (T[i, j, k] - eos.T₀) - eos.βS * (S[i, j, k] - eos.S₀) )

"""
    stability(N², Σ², Pr, Cb)

Return the stability function 

``1 - Cb N^2 / (Pr Σ^2)``

when ``N^2 > 0``, and 1 otherwise.
"""
stability(N², Σ², Pr, Cb::T) where T = min(one(T), max(zero(T), sqrt(one(T) - Cb * N² / (Pr*Σ²))))

"""
    νₑ(ς, Cs, Δ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `Cs`, 
filter with `Δ`, and strain tensor dot product `Σ²`.
"""
νₑ(ς, Cs, Δ, Σ²) = ς * (Cs*Δ)^2 * sqrt(2Σ²)

function ν_ccc(i, j, k, grid, clo::ConstantSmagorinsky, eos, g, u, v, w, T, S)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, g, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    Δ = Δ_ccc(i, j, k, grid, clo)

    ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν_background
end

function ν_ffc(i, j, k, grid, clo::ConstantSmagorinsky, eos, g, u, v, w, T, S)
    N² = ▶xyz_ffc(i, j, k, grid, ∂z_aaf, buoyancy, eos, g, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ffc(i, j, k, grid, u, v, w)
    Δ = Δ_ffc(i, j, k, grid, clo)

    ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν_background
end

function ν_fcf(i, j, k, grid, clo::ConstantSmagorinsky, eos, g, u, v, w, T, S)
    N² = ▶x_faa(i, j, k, grid, ∂z_aaf, buoyancy, eos, g, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_fcf(i, j, k, grid, u, v, w)
    Δ = Δ_fcf(i, j, k, grid, clo)

    ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν_background
end

function ν_cff(i, j, k, grid, clo::ConstantSmagorinsky, eos, g, u, v, w, T, S)
    N² = ▶y_afa(i, j, k, grid, ∂z_aaf, buoyancy, eos, g, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_cff(i, j, k, grid, u, v, w)
    Δ = Δ_cff(i, j, k, grid, clo)

    ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν_background
end

function κ_ccc(i, j, k, grid, clo::ConstantSmagorinsky, eos, g, u, v, w, T, S)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, g, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    Δ = Δ_ccc(i, j, k, grid, clo)

    ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) / clo.Pr + clo.κ_background
end
