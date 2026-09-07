from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, Ellipse, FancyArrowPatch
from scipy.special import j0

OUT = Path(__file__).resolve().parent / 'generated'
OUT.mkdir(exist_ok=True)

ORANGE = '#f26a00'
DEEP_ORANGE = '#d94f00'
DARK = '#202a35'
MID = '#5f6a75'
LIGHT = '#d7dde3'
PALE_BLUE = '#dff1ff'
RED = '#ff3b1f'


def save(fig, name, dpi=320):
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.03)
    plt.close(fig)
    return path


def style_panel(ax, title, title_fs=18):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    card = FancyBboxPatch(
        (0.012, 0.02), 0.976, 0.96,
        boxstyle='round,pad=0.012,rounding_size=0.035',
        facecolor='white', edgecolor='#c9cfd5', linewidth=1.2,
    )
    ax.add_patch(card)
    ax.text(0.04, 0.91, title, ha='left', va='center', fontsize=title_fs,
            color=ORANGE, weight='bold', family='DejaVu Sans')


def build_gaussian_vs_bessel():
    fig, ax = plt.subplots(figsize=(15, 5), dpi=160)
    style_panel(ax, 'GAUSSIAN FOCUS vs BESSEL-LIKE CORE', 18)
    ax.plot([0.5, 0.5], [0.10, 0.82], color=LIGHT, lw=1.2)
    for x0, x1, label in [(0.04, 0.48, 'Gaussian'), (0.53, 0.96, 'Bessel-like')]:
        ax.text((x0 + x1) / 2, 0.78, label, ha='center', va='center', fontsize=16,
                weight='bold', color=DARK)

    # Paraxial Gaussian-beam intensity.
    z = np.linspace(-1, 1, 700)
    x = np.linspace(-0.42, 0.42, 360)
    Z, X = np.meshgrid(z, x)
    zR = 0.19
    w0 = 0.060
    wz = w0 * np.sqrt(1 + (Z / zR) ** 2)
    I = (w0 / wz) ** 2 * np.exp(-2 * X ** 2 / wz ** 2)
    I /= I.max()
    rgba = plt.cm.Oranges(np.clip(I ** 0.55, 0, 1))
    rgba[..., 3] = np.clip(I ** 0.7, 0, 0.82)
    ax.imshow(rgba, extent=(0.06, 0.46, 0.24, 0.68), origin='lower', aspect='auto', interpolation='bilinear')
    zz = np.linspace(-1, 1, 300)
    ww = w0 * np.sqrt(1 + (zz / zR) ** 2)
    ww = ww / ww.max() * 0.16
    zx = 0.26 + zz * 0.20
    ax.plot(zx, 0.46 + ww, color=ORANGE, lw=1.5)
    ax.plot(zx, 0.46 - ww, color=ORANGE, lw=1.5)
    ax.plot([0.055, 0.465], [0.46, 0.46], ls=(0, (5, 4)), color='#7b848c', lw=1.0)
    ax.text(0.26, 0.14, 'short depth of focus', ha='center', va='center', fontsize=14, color=DARK)

    # Finite-energy Bessel-Gauss transverse profile with a finite conical-overlap window.
    z2 = np.linspace(-1, 1, 700)
    x2 = np.linspace(-0.42, 0.42, 360)
    Z2, X2 = np.meshgrid(z2, x2)
    kr = 45.0
    transverse = j0(kr * X2) ** 2 * np.exp(-(X2 / 0.23) ** 2)
    window = 1 / (1 + np.exp(25 * (np.abs(Z2) - 0.62)))
    I2 = transverse * window
    I2 /= I2.max()
    rgba2 = plt.cm.Oranges(np.clip(I2 ** 0.52, 0, 1))
    rgba2[..., 3] = np.clip(I2 ** 0.75, 0, 0.86)
    ax.imshow(rgba2, extent=(0.55, 0.95, 0.24, 0.68), origin='lower', aspect='auto', interpolation='bilinear')
    ax.plot([0.545, 0.955], [0.46, 0.46], ls=(0, (5, 4)), color='#7b848c', lw=1.0)
    for yoff in (0.13, 0.08):
        ax.plot([0.56, 0.67], [0.46 + yoff, 0.46], color=ORANGE, lw=1.35)
        ax.plot([0.56, 0.67], [0.46 - yoff, 0.46], color=ORANGE, lw=1.35)
        ax.plot([0.83, 0.94], [0.46, 0.46 + yoff], color=ORANGE, lw=1.35)
        ax.plot([0.83, 0.94], [0.46, 0.46 - yoff], color=ORANGE, lw=1.35)
    ax.annotate('conical interference\nforms the extended core', xy=(0.72, 0.47), xytext=(0.73, 0.67),
                fontsize=10.5, color=MID, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', lw=1, color=MID, connectionstyle='arc3,rad=0.15'))
    ax.text(0.75, 0.14, 'extended narrow interaction region', ha='center', va='center', fontsize=14, color=DARK)
    ax.text(0.955, 0.055, 'schematic', ha='right', va='center', fontsize=8.5, color='#9199a1')
    return save(fig, '01_gaussian_vs_bessel_like_core.png')


def build_processing_relevance():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    style_panel(ax, 'RELEVANT TO PROCESSING', 17)
    front = Polygon([[0.22, 0.27], [0.62, 0.27], [0.62, 0.67], [0.22, 0.67]], closed=True,
                    facecolor=PALE_BLUE, edgecolor='#63a9d8', alpha=0.55, linewidth=1.2)
    top = Polygon([[0.22, 0.67], [0.62, 0.67], [0.68, 0.72], [0.28, 0.72]], closed=True,
                  facecolor='#eef8ff', edgecolor='#63a9d8', alpha=0.72, linewidth=1.0)
    side = Polygon([[0.62, 0.27], [0.68, 0.32], [0.68, 0.72], [0.62, 0.67]], closed=True,
                   facecolor='#cceaff', edgecolor='#63a9d8', alpha=0.58, linewidth=1.0)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)
    for r, a in [(0.052, 0.16), (0.040, 0.40), (0.027, 0.70)]:
        ax.add_patch(Circle((0.12, 0.47), r, fill=False, edgecolor=RED, linewidth=7 * a, alpha=0.25 + 0.6 * a))
    ax.add_patch(FancyArrowPatch((0.15, 0.47), (0.22, 0.47), arrowstyle='-|>', mutation_scale=14, lw=1.4, color=DARK))
    ax.text(0.12, 0.61, 'Bessel / vortex-Bessel', ha='center', va='center', fontsize=10.5, weight='bold', color=DARK)
    for lw, a, c in [(18, 0.06, RED), (12, 0.10, RED), (7, 0.18, ORANGE), (3, 0.65, '#ffb000')]:
        ax.plot([0.245, 0.60], [0.47, 0.47], lw=lw, color=c, alpha=a, solid_capstyle='round')
    ax.annotate('fused silica', xy=(0.45, 0.70), xytext=(0.42, 0.79), fontsize=10.5, weight='bold', color=DARK,
                arrowprops=dict(arrowstyle='-', color=DARK, lw=1.0), ha='center')
    ax.annotate('elongated optical\ninteraction region', xy=(0.45, 0.47), xytext=(0.49, 0.32), fontsize=10.2,
                weight='bold', color=DARK, arrowprops=dict(arrowstyle='-', color=DARK, lw=1.0), ha='center')
    ax.plot([0.72, 0.72], [0.23, 0.77], color=LIGHT, lw=1.0)
    ax.text(0.75, 0.63,
            'Bessel-like beams can maintain a narrow core over an extended axial range,\nproviding a long optical interaction region inside transparent media.',
            ha='left', va='top', fontsize=12, color=DARK, linespacing=1.35)
    ax.text(0.75, 0.39, 'Motivates studies of:', ha='left', va='top', fontsize=11, color=MID)
    ax.text(0.75, 0.31, 'internal modification  •  waveguides  •  joining', ha='left', va='top', fontsize=11.5,
            color=DARK, weight='bold')
    ax.text(0.04, 0.085, 'Optical-field motivation only; material response is not inferred by this schematic.',
            ha='left', va='center', fontsize=8.5, color='#7d8790')
    return save(fig, '02_processing_relevance.png')


def _perspective_plate(ax, cx, cy, w=0.065, h=0.25, phase='rings', label='SLM'):
    dx = 0.018
    poly = Polygon([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2+0.01], [cx+w/2+dx, cy+h/2], [cx-w/2+dx, cy+h/2-0.01]],
                   closed=True, facecolor='#edf1f5', edgecolor='#4f5b66', lw=1.2)
    ax.add_patch(poly)
    inner = Polygon([[cx-w/2+0.008, cy-h/2+0.018], [cx+w/2-0.007, cy-h/2+0.026],
                     [cx+w/2+dx-0.008, cy+h/2-0.016], [cx-w/2+dx+0.006, cy+h/2-0.024]],
                    closed=True, facecolor='#271b46', edgecolor='#77838f', lw=0.6)
    ax.add_patch(inner)
    if phase == 'rings':
        for rr in np.linspace(0.007, 0.026, 5):
            ax.add_patch(Ellipse((cx+dx/2, cy), 2*rr, rr*0.85, fill=False, edgecolor='#ff4ad8', lw=1.0, alpha=0.9))
        ax.plot([cx+dx/2, cx+dx/2], [cy-0.09, cy+0.09], color='#5ff0ff', lw=0.8, alpha=0.8)
    else:
        for xx in np.linspace(cx-w/2+0.014, cx+w/2+dx-0.014, 6):
            ax.plot([xx, xx+0.008], [cy-h/2+0.04, cy+h/2-0.04], color='#ff4ad8', lw=1.1)
    ax.text(cx+dx/2, cy-h/2-0.04, label, ha='center', va='top', fontsize=11, weight='bold', color=DARK)


def build_optical_route():
    fig, ax = plt.subplots(figsize=(16, 5), dpi=160)
    style_panel(ax, 'EXPERIMENTAL / MODELLED OPTICAL ROUTE', 17)
    y = 0.45
    xline = np.linspace(0.08, 0.94, 800)
    for lw, a in [(18, 0.025), (10, 0.055), (4, 0.16), (1.6, 0.9)]:
        ax.plot(xline, np.full_like(xline, y), color=RED, lw=lw, alpha=a, solid_capstyle='round', zorder=0)
    laser = FancyBboxPatch((0.04, 0.36), 0.075, 0.18, boxstyle='round,pad=0.005,rounding_size=0.01',
                           facecolor='#1e252c', edgecolor='#111820', lw=1.2, zorder=3)
    ax.add_patch(laser)
    ax.add_patch(Circle((0.113, y), 0.018, facecolor='#2e3944', edgecolor='#111820', lw=1.0, zorder=4))
    ax.add_patch(Circle((0.113, y), 0.009, facecolor='#ff5a1f', edgecolor='#ffd3c2', lw=0.8, zorder=5))
    ax.text(0.075, 0.29, 'fs laser', ha='center', va='top', fontsize=11, weight='bold', color=DARK)
    _perspective_plate(ax, 0.23, y, w=0.066, h=0.26, phase='rings', label='SLM 1')
    _perspective_plate(ax, 0.39, y, w=0.066, h=0.26, phase='grating', label='SLM 2')
    ax.text(0.24, 0.73, 'conditioning / correction', ha='center', va='bottom', fontsize=9.5, color=MID)
    ax.text(0.40, 0.73, 'axicon + vortex + carrier', ha='center', va='bottom', fontsize=9.5, color=MID)
    for xc, lab in [(0.56, 'L1'), (0.72, 'L2')]:
        ax.add_patch(Ellipse((xc, y), 0.030, 0.29, facecolor='#dff3ff', edgecolor='#5793bd', lw=1.2, alpha=0.65, zorder=4))
        ax.add_patch(Ellipse((xc-0.003, y), 0.010, 0.25, facecolor='white', edgecolor='none', alpha=0.32, zorder=5))
        ax.text(xc, 0.29, lab, ha='center', va='top', fontsize=11, weight='bold', color=DARK)
    ax.plot([0.64, 0.64], [0.31, 0.66], ls=(0, (4, 3)), color='#8a949d', lw=1.0)
    ax.add_patch(Rectangle((0.626, 0.365), 0.028, 0.17, facecolor='#e8eef2', edgecolor='#59636d', lw=0.9, zorder=4))
    ax.add_patch(Circle((0.64, y), 0.012, facecolor='#1f252b', edgecolor='#080b0f', lw=0.8, zorder=5))
    ax.text(0.64, 0.72, 'Fourier plane\n+1 order selected', ha='center', va='bottom', fontsize=9.5, color=MID)
    ax.plot([0.56, 0.72], [0.25, 0.25], color='#5f6a75', lw=1.0)
    ax.plot([0.56, 0.56], [0.25, 0.27], color='#5f6a75', lw=1.0)
    ax.plot([0.72, 0.72], [0.25, 0.27], color='#5f6a75', lw=1.0)
    ax.text(0.64, 0.22, '4f spatial filter', ha='center', va='top', fontsize=10.5, weight='bold', color=DARK)
    axicon = Polygon([[0.80, 0.34], [0.80, 0.56], [0.86, y]], closed=True, facecolor='#d7efff',
                     edgecolor='#5085ac', lw=1.2, alpha=0.72, zorder=4)
    ax.add_patch(axicon)
    ax.add_patch(Ellipse((0.80, y), 0.025, 0.22, facecolor='#edf9ff', edgecolor='#6aa6d1', lw=1.0, alpha=0.8, zorder=5))
    ax.text(0.83, 0.29, 'physical axicon', ha='center', va='top', fontsize=10.5, weight='bold', color=DARK)
    ax.text(0.83, 0.71, 'conical refraction', ha='center', va='bottom', fontsize=9.5, color=MID)
    ax.add_patch(Rectangle((0.925, 0.34), 0.040, 0.22, facecolor='#f8f9fa', edgecolor='#4f5964', lw=1.2, zorder=4))
    for rr, c, lw in [(0.013, '#f08a24', 2.0), (0.007, '#ffbb42', 1.4)]:
        ax.add_patch(Ellipse((0.945, y), rr*2, rr*2*1.6, fill=False, edgecolor=c, lw=lw, zorder=6))
    ax.text(0.945, 0.29, 'camera / sample', ha='center', va='top', fontsize=10.5, weight='bold', color=DARK)
    for xa, xb in [(0.12,0.20), (0.27,0.36), (0.43,0.54), (0.58,0.62), (0.66,0.70), (0.74,0.79), (0.87,0.92)]:
        ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle='-|>', mutation_scale=12, lw=1.1,
                                     color=DEEP_ORANGE, zorder=7))
    ax.text(0.50, 0.10,
            'Physical order used by the component-resolved numerical route; presentation schematic, not a scale drawing.',
            ha='center', va='center', fontsize=9.2, color='#747f88')
    return save(fig, '03_experimental_modelled_optical_route.png')


def build_reference_field():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    style_panel(ax, 'VORTEX-BESSEL REFERENCE FIELD', 17)
    ax.plot([0.04, 0.96], [0.82, 0.82], color=ORANGE, lw=1.5)
    eq = r'$U_{\ell}(r,\phi,0)=A_0 J_{\ell}(k_r r)\,\exp\!\left(-\frac{r^2}{w_0^2}\right)\,e^{i\ell\phi}$'
    ax.text(0.5, 0.58, eq, ha='center', va='center', fontsize=29, color='#11151a')
    ax.text(0.5, 0.34, 'Clean reference field before realistic component errors are added.', ha='center', va='center',
            fontsize=12.5, color=MID)
    ax.text(0.5, 0.23,
            r'Changing topological charge $\ell$ controls the dark-core size and transverse energy distribution.',
            ha='center', va='center', fontsize=12.5, color=MID)
    return save(fig, '04_vortex_bessel_reference_field.png')


def main():
    for builder in (build_gaussian_vs_bessel, build_processing_relevance, build_optical_route, build_reference_field):
        print(builder())


if __name__ == '__main__':
    main()
