#!/usr/bin/env python
"""
Quenched ceramic slab.

The slab of Tang et al. (Int. J. Solids Struct. 80 (2016) 520-531), quenched into water as in
the experiment of Shao et al. (2011) and the computation of Bourdin et al. (2014). The cooled
faces contract and go into tension, and an array of cracks nucleates along them, of which a
subset propagates into the depth while the rest arrest. There is no mechanical load.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    A slab of 50 x 10 mm. One quarter, 25 x 5 mm, is computed with symmetry conditions on the
    two cuts, so its long face and its end face are the cooled faces.

Material
    Alumina under plane stress: E = 370 GPa, nu = 0.30, Gc = 42.47 J/m^2,
    alpha = 7.5e-6 1/K, k = 31 W/(m K), rho = 3980 kg/m^3, c = 880 J/(kg K). The domain is
    divided into square cells of 0.025 mm, and each cell draws its modulus and its strength from
    a Weibull distribution of modulus 15 about E and 180 MPa. The strength enters through the
    toughness, Gc_cell = (256/27) l sigma_t^2 / E, and the toughness field is rescaled so that
    its mean is Gc. The conductivity is not degraded. Regularization length l = 0.05 mm.

Boundary conditions
    Thermal: the slab starts uniform at T0 and its cooled faces are held at the bath
    temperature 293 K. The symmetry cuts are insulated.
    Mechanical: symmetry conditions on the cuts, all other faces traction free.

Loading
    100 load steps on the clock t_n = t_s + (t_f - t_s) ((n - 1)/99)^3 from t_s = 1e-4 s to
    t_f = 0.2 s, after the uniform state at t = 0.

Command line
    --T0              initial temperature in K (default 873). The bath is --T_env 293 K, so
                      the thermal load is T0 - 293 K
    --dT_final        must be set to 293 - T0 (default -580). It scales the temperature
                      network and does not move the bath
    --het_seed        realization of the Weibull field (default 1)
    --seed            seed of the network initialization (default 0)
    --bc_T robin      convective cooled faces instead of a prescribed temperature
    --resume          continue from the latest checkpoint in the output folder
    --out             output folder (default runs/slab_dT580)
    Every solver option is available, see --help.

    The four thermal loads of the paper, each on its own realization of the field:

    python examples/slab.py --T0 573 --dT_final -280 --het_seed 2 --out runs/slab_dT280
    python examples/slab.py --T0 673 --dT_final -380 --het_seed 3 --out runs/slab_dT380
    python examples/slab.py --T0 773 --dT_final -480 --het_seed 2 --out runs/slab_dT480
    python examples/slab.py                                       # 580 K, field 1

    The repeats at 380 K add --seed 1 and --seed 2.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    strip="50,10",
    quarter=1,                      # one quarter, symmetry planes on the cuts
    l=0.05,
    sigma_t=180.0, het_m=15.0, het_cell=0.025, het_seed=1,
    gc_mean=0.04247,                # rescales the toughness field to this mean
    T0=873.0, dT_final=-580.0,
    bc_T="dirichlet_net",           # prescribed temperature on the cooled faces
    levels="64,192,384",
    levels_T="64,128,384",
    grid_n=384,
    N=80000,
    N_boundary=8000,
    nsteps=100,
    iters=2400,
    iters0=1500,
    iters1=3000,                    # the first transient step
    iters_T=800,
    iters_T1=1600,
    ir_tol=0.0,
    gamma_ir=3.6e4,
    ir_thresh=0.5,                  # the penalty acts on the crack set phi >= 1/2
    lr_final=5e-5,                  # learning rate decayed to this value within each step
    adam_persist=1,                 # optimizer moments carried over between load steps
    ckpt_every=1,
    plot_every=5,
    out="runs/slab_dT580",
)


if __name__ == "__main__":
    run_cli("quench", description=__doc__, defaults=DEFAULTS)
