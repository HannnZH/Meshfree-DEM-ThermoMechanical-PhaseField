#!/usr/bin/env python
"""
Single-edge-notched shear test of a functionally graded plate under thermal load.

The notched plate of sen_tension.py loaded in shear, on a material graded linearly in x with
the end values of the released code of Tangella et al. (2022). The top edge is displaced
horizontally while its temperature is ramped, and the bottom edge is held at the reference
temperature.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    Unit square [0, 1] x [0, 1] with a horizontal edge notch from (0, 0.5) to (0.5, 0.5).

Material
    Plane strain, graded linearly from x = 0 to x = 1: E from 340 to 450 GPa, Gc0 from 42.47
    to 120 J/m^2, k from 300 to 350 W/(m K), and rho from 2450 to 4000 kg/m^3, with nu = 0.22,
    alpha = 8.0e-6 1/K and c = 775 J/(kg K) constant. The heat capacity c and the time scale are
    both divided by 1e3, as in sen_tension.py. The fracture energy carries the temperature law of Ruan et al.
    (2023) with b1 = 1.80, b2 = 1.10, T_ref = 300 K and T_max = 1000 K, and the conductivity
    is degraded with the phase field.

Boundary conditions
    Thermal: the bottom edge at T0 = 300 K, the top edge at T0 + DeltaT, the other edges and the
    notch faces adiabatic.
    Mechanical: the bottom edge fixed, the top edge displaced horizontally by delta.

Loading
    160 load steps of 2.5e-5 mm. DeltaT grows by 2.5 K per load step to dT_final and is held
    from then on.

Command line
    --dT_final  final temperature change of the top edge in K (default 30)
    --seed      seed of the network initialization (default 0)
    --resume    continue from the latest checkpoint in the output folder
    --out       output folder (default runs/sen_shear_fgm)
    Every solver option is available, see --help.

    python examples/sen_shear_fgm.py                                         # heated by 30 K
    python examples/sen_shear_fgm.py --dT_final -30 --out runs/sen_shear_fgm_m30   # cooled
    python examples/sen_shear_fgm.py --dT_final 0 --out runs/sen_shear_fgm_iso     # isothermal

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="shear",
    heat_on=True,
    fgm=True,
    # the material at x = 0; the values at x = 1 are set by the graded problem
    E=340e3, nu=0.22, Gc=0.04247, alpha=8.0e-6,
    kcond=300.0, rho_si=2450.0, cp_si=0.775,
    T0=300.0, dT_final=30.0,
    stress_state="plane_strain",
    alpha_mode="ps3d",
    gc_tdep=1, gc_b1=1.80, gc_b2=1.10, gc_Tref=300.0,
    kdeg=1,
    l=0.01,
    levels="32,128,384",
    N=12000,
    nsteps=160,
    delta_max=4.0e-3,               # 2.5e-5 mm per load step
    iters=2000,
    iters0=1000,
    iters1=4000,
    iters_T1=1200,
    ckpt_every=1,
    plot_every=10,
    out="runs/sen_shear_fgm",
)


if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
