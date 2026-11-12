#!/usr/bin/env python
"""
Directionally quenched glass strip.

The experiment of Yuse and Sano (Nature 362 (1993) 329-331), set up as in the phase-field
study of Zeng and Dolbow (2026). A hot glass strip is drawn at constant speed into a cold bath
and a crack follows the immersion front. As the temperature jump grows, the crack runs straight,
oscillates, forks, and at larger jumps several cracks start along the leading edge.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    A strip of 30 x 16 mm (half-width b = 8 mm) with an edge notch of length 5 mm at the
    leading edge x = 0, offset 0.25 mm above the centerline.

Material
    Glass under plane stress: E = 70 GPa, nu = 0.22, Gc = 8.25 J/m^2, alpha = 7.7e-6 1/K,
    thermal diffusivity D = 0.47 mm^2/s.

Temperature
    Not computed. The steady profile of the moving front is imposed in closed form,
        T(x, t) = T_bath + dT [1 - exp(-(V/D) <x - V t>)],
    with the bath at 296 K, the drawing speed V = 0.4 mm/s and <.> the positive part.

Boundary conditions
    u = 0 along the trailing edge x = 30 mm and v = 0 at its midpoint, which removes the rigid
    modes. All other edges are traction free, so the thermal strain is the only load.

Loading
    The front travels 24 mm in 240 load steps of 0.1 mm, with 5000 optimizer iterations per
    step and 10000 at the first.

Command line
    --dT        temperature jump across the front in K (default 165)
    --seed      seed of the network initialization (default 0)
    --resume    continue from the latest checkpoint in the output folder
    --out       output folder (default runs/strip_dT165)
    Every solver option is available, see --help.

    python examples/strip.py                                   # 165 K, oscillating crack
    python examples/strip.py --dT 130 --out runs/strip_dT130   # straight crack
    python examples/strip.py --dT 217 --out runs/strip_dT217   # forking crack
    python examples/strip.py --dT 307 --out runs/strip_dT307   # several cracks

    The paper also reports 145, 155, 170, 175 and 190 K, which differ in --dT only.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    dT=165.0,
    l=0.1,
    levels="64,256,624",            # side lengths along y; the x side scales with the aspect ratio
    grid_n=384,
    N=100000,
    nsteps=240,                     # 0.1 mm of front travel per load step
    iters=5000,
    iters0=10000,                   # the first step solves the fully developed front
    gamma_ir=250.0 * 8.25e-3 / 0.1,  # irreversibility penalty 250 Gc / l
    ir_tol=0.0,
    ir_thresh=0.5,                  # the penalty acts on the crack set phi >= 1/2
    lr_final=5e-5,                  # learning rate decayed to this value within each step
    adam_persist=1,                 # optimizer moments carried over between load steps
    ckpt_every=1,
    plot_every=2,
    out="runs/strip_dT165",
)


if __name__ == "__main__":
    run_cli("yusesano", description=__doc__, defaults=DEFAULTS)
