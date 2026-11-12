#!/usr/bin/env python
"""
Single-edge-notched tension test under thermal load.

The thermoelastic single-edge-notched benchmark of Tangella et al. (2022). The top edge is
pulled vertically while its temperature is ramped, and the bottom edge is held at the reference
temperature. The thermal load changes the peak load by only a few percent, and the direction of
the change differs between implementations.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    Unit square [0, 1] x [0, 1] with a horizontal edge notch from (0, 0.5) to (0.5, 0.5).

Material
    Homogeneous, plane strain: E = 340 GPa, nu = 0.22, alpha = 8.0e-6 1/K, k = 300 W/(m K),
    rho = 2450 kg/m^3, c = 775 J/(kg K). The heat capacity c and the time scale are both
    divided by 1e3, which leaves every field unchanged since the thermal load is tied to the
    load step. The fracture
    energy follows the law of Ruan et al. (2023),
        Gc(T) = Gc0 [1 - b1 (T - T_ref)/T_max + b2 ((T - T_ref)/T_max)^2],
    with Gc0 = 42.47 J/m^2, b1 = 1.80, b2 = 1.10, T_ref = 300 K and T_max = 1000 K. The
    conductivity is degraded with the phase field.

Boundary conditions
    Thermal: the bottom edge at T0 = 300 K, the top edge at T0 + DeltaT, the other edges and the
    notch faces adiabatic.
    Mechanical: the bottom edge fixed, the top edge displaced vertically by delta.

Loading
    80 load steps of 1e-5 mm. DeltaT grows by 2.5 K per load step over the first 20 steps to
    dT_final and is held from then on.

Command line
    --dT_final  final temperature change of the top edge in K (default 50)
    --seed      seed of the network initialization (default 0)
    --resume    continue from the latest checkpoint in the output folder
    --out       output folder (default runs/sen_tension)
    Every solver option is available, see --help.

    python examples/sen_tension.py                                       # heated by 50 K
    python examples/sen_tension.py --dT_final -50 --out runs/sen_tension_m50   # cooled
    python examples/sen_tension.py --dT_final 0 --out runs/sen_tension_iso     # isothermal

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="tension",
    heat_on=True,
    E=340e3, nu=0.22, Gc=0.04247, alpha=8.0e-6,
    kcond=300.0, rho_si=2450.0, cp_si=0.775,
    T0=300.0, dT_final=50.0,
    # 2.5 K per load step on the clock t = time_per_delta * delta:
    # ramp_rate * time_per_delta * (delta per step) = 2.5e8 * 1e-3 * 1e-5 = 2.5 K
    ramp_rate=2.5e8,
    stress_state="plane_strain",
    alpha_mode="ps3d",
    gc_tdep=1, gc_b1=1.80, gc_b2=1.10, gc_Tref=300.0,
    kdeg=1,
    l=0.01,
    levels="32,128,384",
    N=12000,
    nsteps=80,
    delta_max=8.0e-4,               # 1e-5 mm per load step
    iters=2000,
    iters0=1000,
    iters1=4000,
    iters_T1=1200,
    ckpt_every=1,
    plot_every=10,
    out="runs/sen_tension",
)


if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
