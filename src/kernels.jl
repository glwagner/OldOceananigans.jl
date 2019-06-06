function ▶z_buoyancy_aaf(i, j, k, grid::Grid{FT}, eos, grav, T, S) where FT
    if k == 1
        return buoyancy(i, j, 1, grid, eos, grav, T, S)
    else
        return FT(0.5) * (buoyancy(i, j, k, grid, eos, grav, T, S)
                        + buoyancy(i, j, k-1, grid, eos, grav, T, S))
    end
end

function ▶z_buoyancy_w_aaf(i, j, k, grid::Grid{FT}, eos, grav, T, S) where FT
    if k == 1
        return -zero(FT)
    else
        return FT(0.5) * (buoyancy(i, j, k, grid, eos, grav, T, S)
                        + buoyancy(i, j, k-1, grid, eos, grav, T, S))
    end
end



"""
    update_hydrostatic_pressure!(ph, grid, constants, eos, T, S)
Calculate the perbutation hydrostatic pressure `ph` from the buoyancy field
associated with temperature `T`, salinity `S`, gravitational constant
`constants.g`, and equation of state `eos`.

The perturbation hydrostatic pressure `ph` is defined as the part of pressure
that balances buoyancy in the vertical momentum equation,

    `0 = -∂z ph + b`.

Pressure and buoyancy are both are defined at cell centers.
Thus evaluting the discrete hydrostatic pressure equation on face `k`
requires interpolating the buoyancy field. Given the reverse indexing convention,
the hydrostatic pressure gradient on face `k` is `(phᵏ⁻¹ - phᵏ) / Δz`.
The discrete hydrostatic pressure equation is therefore:

    `0 = -(phᵏ⁻¹ - phᵏ) / Δz + (bᵏ + bᵏ⁻¹) / 2`,

which, rearranged and using the notation `▶z_aaf(bᵏ) = (bᵏ + bᵏ⁻¹) / 2`,
yields

    `pᵏ = pᵏ⁻¹ - Δz * ▶z_aaf(bᵏ)`.

We solve this discrete equation by integrating from the top down,
using the surface buoyancy to set the boundary condition.
"""
function update_hydrostatic_pressure!(pHY′, grid::Grid{FT}, constants, eos, T, S) where FT
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = - grid.Δz * ▶z_buoyancy_aaf(i, j, 1, grid, eos, constants.g, T, S)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] - grid.Δz * ▶z_buoyancy_aaf(i, j, k, grid, eos, constants.g, T, S)
            end
        end
    end

    @synchronize
end

"""Store previous source terms before updating them."""
function update_previous_source_terms!(grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gpu[i, j, k] = Gu[i, j, k]
                @inbounds Gpv[i, j, k] = Gv[i, j, k]
                @inbounds Gpw[i, j, k] = Gw[i, j, k]
                @inbounds GpT[i, j, k] = GT[i, j, k]
            end
        end
    end
    @synchronize
end

"Store previous value of the source term and calc current source term."
function calc_u_source_term!(grid, constants, eos, closure, pHY′, u, v, w, T, S, Gu, diffusivities, F, iter)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # u-momentum equation
                @inbounds Gu[i, j, k] = (-u∇u(grid, u, v, w, i, j, k)
                                            + fv(grid, v, constants.f, i, j, k)
                                            - δx_c2f(grid, pHY′, i, j, k) / grid.Δx
                                            + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)
                                            + F.u(grid.xF[i], grid.yC[j], grid.zC[k], u, v, w, T, S, grid, i, j, k, iter)
                                        )
            end
        end
    end

    @synchronize
end


"Store previous value of the source term and calc current source term."
function calc_v_source_term!(grid, constants, eos, closure, pHY′, u, v, w, T, S, Gv, diffusivities, F, iter)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # v-momentum equation
                @inbounds Gv[i, j, k] = (-u∇v(grid, u, v, w, i, j, k)
                                            - fu(grid, u, constants.f, i, j, k)
                                            - δy_c2f(grid, pHY′, i, j, k) / grid.Δy
                                            + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)
                                            + F.v(grid.xC[i], grid.yF[j], grid.zC[k], u, v, w, T, S, grid, i, j, k, iter)
                                        )
            end
        end
    end

    @synchronize
end


"Store previous value of the source term and calc current source term."
function calc_w_source_term!(grid, constants, eos, closure, pHY′, u, v, w, T, S, Gw, diffusivities, F, iter)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # w-momentum equation
                @inbounds Gw[i, j, k] = (-u∇w(grid, u, v, w, i, j, k)
                                         # + ▶z_buoyancy_aaf(i, j, k, grid, eos, grav, T, S)
                                         + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)
                                         + F.w(grid.xC[i], grid.yC[j], grid.zF[k], u, v, w, T, S, grid, i, j, k, iter)
                                        )
            end
        end
    end

    @synchronize
end

"Store previous value of the source term and calc current source term."
function calc_T_source_term!(grid, constants, eos, closure, pHY′, u, v, w, T, S, GT, diffusivities, F, iter)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

                # temperature equation
                @inbounds GT[i, j, k] = (-div_flux(grid, u, v, w, T, i, j, k)
                                         + ∇_κ_∇T(i, j, k, grid, T, closure, diffusivities)
                                         + F.T(grid.xC[i], grid.yC[j], grid.zC[k], u, v, w, T, S, grid, i, j, k, iter)
                                        )
            end
        end
    end

    @synchronize
end

function adams_bashforth_update_source_terms!(grid::Grid{FT}, Gu, Gv, Gw, GT, GS,
                                              Gpu, Gpv, Gpw, GpT, GpS, χ) where FT
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gu[i, j, k] = (FT(1.5) + χ)*Gu[i, j, k] - (FT(0.5) + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (FT(1.5) + χ)*Gv[i, j, k] - (FT(0.5) + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (FT(1.5) + χ)*Gw[i, j, k] - (FT(0.5) + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (FT(1.5) + χ)*GT[i, j, k] - (FT(0.5) + χ)*GpT[i, j, k]
            end
        end
    end
    @synchronize
end

"Store previous value of the source term and calc current source term."
function calc_poisson_right_hand_side!(::CPU, grid, Δt, u, v, w, Gu, Gv, Gw, RHS)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end

    @synchronize
end

function calc_poisson_right_hand_side!(::GPU, grid, Δt, u, v, w, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
                # which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                end
            end
        end
    end

    @synchronize
end

function idct_permute!(grid, ϕ, pNHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                #if k > Nz/2
                if k <= Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end

    @synchronize
end

function update_velocities_and_tracers!(grid, u, v, w, T, S, pNHS, Gu, Gv, Gw,
                                        GT, GS, Gpu, Gpv, Gpw, GpT, GpS, Δt)

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(grid, pNHS, i, j, k) / grid.Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(grid, pNHS, i, j, k) / grid.Δy)) * Δt
                @inbounds T[i, j, k] = T[i, j, k] + (GT[i, j, k] * Δt)
                #@inbounds S[i, j, k] = S[i, j, k] + (GS[i, j, k] * Δt)
            end
        end
    end

    @synchronize
end

"Compute the vertical velocity w from the continuity equation."
function compute_w_from_continuity!(grid, u, v, w)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds w[i, j, k] = w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, u, v)
            end
        end
    end

    @synchronize
end
