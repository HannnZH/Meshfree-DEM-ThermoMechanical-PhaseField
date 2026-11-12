#!/usr/bin/env python
"""
Quenched ceramic annulus.

The quench of the slab carried onto a curved domain. An annulus of the slab's ceramic is
quenched through both rims, and arrays of cracks nucleate along each rim and grow inward from
the outer rim and outward from the bore. There is no mechanical load.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    An annulus of inner radius 2.5 mm and outer radius 10 mm. One quarter is computed on a
    single rational quadratic NURBS patch that represents both circles exactly, with symmetry
    conditions on the two radial cuts.

Material
    The alumina of slab.py under plane stress, with the same Weibull field construction on
    cells of 0.025 mm and l = 0.05 mm. The conductivity is not degraded.

Boundary conditions
    Thermal: the annulus starts uniform at T0 and both rims are held at the bath temperature
    293 K. The radial cuts are insulated.
    Mechanical: symmetry conditions on the cuts, both rims traction free.

Loading
    The clock of slab.py: 100 load steps from 1e-4 s to 0.2 s, concentrated in the early
    transient.

Command line
    --T0              initial temperature in K (default 873). The bath is --T_env 293 K, so
                      the thermal load is T0 - 293 K
    --dT_final        must be set to 293 - T0 (default -580). It scales the temperature
                      network and does not move the bath
    --het_seed        realization of the Weibull field (default 1)
    --seed            seed of the network initialization (default 0)
    --resume          continue from the latest checkpoint in the output folder
    --out             output folder (default runs/annulus_dT580)
    Every solver option is available, see --help.

    The four thermal loads of the paper, all on field 1:

    python examples/annulus.py --T0 573 --dT_final -280 --out runs/annulus_dT280
    python examples/annulus.py --T0 673 --dT_final -380 --out runs/annulus_dT380
    python examples/annulus.py --T0 773 --dT_final -480 --out runs/annulus_dT480
    python examples/annulus.py                                     # 580 K

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    ring_ri=2.5, ring_ro=10.0,
    ring_quarter=True,              # one quarter, symmetry planes on the radial cuts
    l=0.05,
    het_cell=0.025, het_seed=1,
    gc_mean=0.04247,                # rescales the toughness field to this mean
    bc_T="dirichlet_net",           # prescribed temperature on both rims
    levels="64,256,1280",           # side lengths along the arc; the radial side scales
    levels_T="64,256,320",
    grid_n=320,
    N=80000,
    N_boundary=8000,
    nsteps=100,
    iters=2400,
    iters0=1500,
    iters1=3000,                    # the first transient step
    iters_T=800,
    iters_T1=1600,
    gamma_ir=3.6e4,
    ir_tol=0.0,
    ir_thresh=0.5,                  # the penalty acts on the crack set phi >= 1/2
    lr_final=5e-5,                  # learning rate decayed to this value within each step
    adam_persist=1,                 # optimizer moments carried over between load steps
    ckpt_every=1,
    plot_every=2,
    out="runs/annulus_dT580",
)


if __name__ == "__main__":
    run_cli("ringquench", description=__doc__, defaults=DEFAULTS)
