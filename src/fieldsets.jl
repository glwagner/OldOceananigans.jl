function VelocityFields(arch::Architecture, grid::Grid)
    u = FaceFieldX(arch, grid)
    v = FaceFieldY(arch, grid)
    w = FaceFieldZ(arch, grid)
    return (u=u, v=v, w=w)
end

function TracerFields(arch::Architecture, grid::Grid)
    θ = CellField(arch, grid)  # Temperature θ to avoid conflict with type T.
    S = CellField(arch, grid)
    return (T=θ, S=S)
end

function PressureFields(arch::Architecture, grid::Grid)
    pNHS = CellField(arch, grid)
    return (pHY′=nothing, pNHS=pNHS)
end

function SourceTerms(arch::Architecture, grid::Grid)
    Gu = FaceFieldX(arch, grid)
    Gv = FaceFieldY(arch, grid)
    Gw = FaceFieldZ(arch, grid)
    GT = CellField(arch, grid)
    GS = CellField(arch, grid)
    return (Gu=Gu, Gv=Gv, Gw=Gw, GT=GT, GS=GS)
end

function StepperTemporaryFields(arch::Architecture, grid::Grid{T}) where T
    fC1 = CellField(arch, grid)

    # Forcing Float64 for these fields as it's needed by the Poisson solver.
    fCC1 = CellField(Complex{T}, arch, grid)
    fCC2 = CellField(Complex{T}, arch, grid)

    return (fC1=fC1, fCC1=fCC1, fCC2=fCC2)
end
