using Oceananigans, Cthulhu

model = Model(N=(32, 32, 32), L=(1, 1, 1),
              closure=ConstantSmagorinsky()) 

@descend time_step!(model, 1, 1)
