"""Run beside the original validation script and data. No new trajectories needed."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import run_single_reference_validation as v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=float, default=-0.11)
    args = parser.parse_args()
    v.x_ref_A = args.reference
    data = np.load(Path(__file__).with_name('two_region_validation_data.npz'))
    out = Path(__file__).with_name('six_panel_output')
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for col, (case, (A1, A2)) in enumerate(v.cases.items()):
        xa = np.linspace(v.b1, v.a, len(data[f'{case}_PstA']))
        xb = np.linspace(v.a, v.b2, len(data[f'{case}_PstB']))
        ua = v.reconstruct_region_A_shape(xa, data[f'{case}_PstA'], data[f'{case}_mfptA'])
        ub = v.reconstruct_region_B_shape(xb, data[f'{case}_PstB'], data[f'{case}_mfptB'])
        tm = v.build_tm_shapes(A1, A2)
        ref = float(v.beta_U(args.reference, A1, A2))
        for row, (method, xA, UA, xB, UB) in enumerate([
            ('RW', xa[1:-1], ua, xb[1:-1], ub),
            ('TM', *tm),
        ]):
            result = v.single_reference_match(xA, UA, xB, UB, ref)
            ax = axes[row, col]
            xx = np.linspace(v.b1, v.b2, 600)
            ax.plot(xx, v.beta_U(xx, A1, A2), color='black', label='Model')
            ax.plot(xA, result['UA'], '--', color='tab:blue', label=f'{method} A')
            ax.plot(xB, result['UB'], '--', color='tab:orange', label=f'{method} B matched')
            ax.axvline(v.a, color='gray', ls=':', lw=1)
            ax.plot(args.reference, ref, 'o', color='tab:blue', ms=4)
            ax.plot(v.a, result['Ua_fit'], 'x', color='tab:red', ms=6)
            err = abs(result['Ua_fit'] - float(v.beta_U(v.a, A1, A2)))
            print(f'{case:6s} {method}: beta U(a)={result["Ua_fit"]:.9f}, absolute deviation={err:.9f}')
            ax.set_title(f'({chr(97+row*3+col)}) {case.capitalize()} — {method}')
            ax.set_ylabel(r'$\beta U(x)$')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel('x')
            np.savetxt(out / f'{case}_{method}_A.csv', np.column_stack([xA, result['UA']]), delimiter=',', header='x,beta_U', comments='')
            np.savetxt(out / f'{case}_{method}_B.csv', np.column_stack([xB, result['UB']]), delimiter=',', header='x,beta_U', comments='')
    fig.suptitle(f'Single-reference reconstruction: x_R={args.reference}, six-point quadratic matching')
    fig.tight_layout()
    fig.savefig(out / 'reconstruction_six_panels.png', dpi=220)
    fig.savefig(out / 'reconstruction_six_panels.pdf')
    print(f'Figures and reconstructed profiles: {out}')

if __name__ == '__main__':
    main()
