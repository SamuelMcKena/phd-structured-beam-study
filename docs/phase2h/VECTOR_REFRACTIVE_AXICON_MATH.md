# Phase 2H — vector two-surface refractive axicon mathematics

## Scope

This stage models a macroscopic plano-conical refractive axicon as two physical
dielectric interfaces and constructs a complex vector boundary field on a fixed
laboratory plane immediately downstream of the optic.  Diffraction after that
plane is handled by the repository vector angular-spectrum / Debye machinery.

It is a vector geometrical-optics/eikonal boundary-field model, not a full-volume
FDTD/FEM solution.  The intended asymptotic regime is a macroscopic optic whose
surface radii/thicknesses are many wavelengths, with a coherent field that can be
sampled on the entrance surface.

Primary literature motivating the formulation:

- Yun, Crabtree & Chipman, *Applied Optics* **50**, 2855–2865 (2011): 3-D
  polarization ray tracing, including refraction and diattenuation.
- Kim et al., *JOSA A* **35**, 526–535 (2018): vectorial diffraction with both ray
  vectors and electromagnetic field vectors traced to generate aperture fields.
- Bin & Zhu, *Applied Optics* **37**, 2563–2568 (1998): oblique axicon diffraction,
  theory checked experimentally.
- Thaning, Jaroszewicz & Friberg, *Applied Optics* **42**, 9–17 (2003): broadened
  focal lines / caustics under oblique axicon illumination, checked by diffraction
  simulation and experiment.

## 1. Coordinate systems and rigid pose

Let primed coordinates be fixed to the axicon.  The rotation

\[
\mathbf r_{\rm lab}=R\,\mathbf r_{\rm ax}
\]

uses the repository convention

\[
R=R_y(\theta_y)R_x(\theta_x).
\]

The field is sampled on the physical flat entrance surface.  Its complex vector
components remain in the fixed laboratory basis during the rotated-plane sample;
for interface calculations they are transformed into the axicon basis by

\[
\mathbf E_{\rm ax}=R^T\mathbf E_{\rm lab}.
\]

A lateral axicon decentre is a tangent-plane displacement and is therefore kept
separate from rigid rotation.

## 2. Physical plano-conical geometry

For the current flat-first geometry,

\[
z'=0
\]

is the air-to-glass surface and

\[
z'=t_c-r'\tan\gamma,
\qquad
r'=\sqrt{x'^2+y'^2}
\]

is the glass-to-air conical surface.  The physical edge thickness is

\[
t_e=t_c-R_c\tan\gamma>0,
\]

where \(R_c\) is the declared clear radius.  The code refuses impossible
centre-thickness/base-angle/clear-radius combinations.

The outward normal from glass to air at the conical surface is

\[
\hat{\mathbf n}_2=
\begin{bmatrix}
\sin\gamma\cos\phi\\
\sin\gamma\sin\phi\\
\cos\gamma
\end{bmatrix}.
\]

## 3. Exact ray/surface intersection

An internal ray is

\[
\mathbf r'(s)=\mathbf r'_0+s\hat{\mathbf s}_g.
\]

Substitution into the cone equation gives a quadratic in \(s\).  The solver
selects the first positive physical root and rejects degenerate/missing
intersections.  This determines the actual glass path length for every entrance
sample rather than assigning a thin phase delay.

## 4. Vector Snell law

At any interface, let the unit normal point from medium 1 to medium 2.  Decompose

\[
\hat{\mathbf s}_1=
(\hat{\mathbf s}_1\cdot\hat{\mathbf n})\hat{\mathbf n}
+\hat{\mathbf s}_{1t}.
\]

Tangential wave-vector continuity is

\[
n_1\hat{\mathbf s}_{1t}=n_2\hat{\mathbf s}_{2t}.
\]

Hence

\[
\hat{\mathbf s}_{2t}=\frac{n_1}{n_2}\hat{\mathbf s}_{1t},
\]

and the transmitted unit direction is

\[
\boxed{
\hat{\mathbf s}_2=
\frac{n_1}{n_2}\hat{\mathbf s}_{1t}
+\sqrt{1-\left\|\frac{n_1}{n_2}\hat{\mathbf s}_{1t}\right\|^2}\,
\hat{\mathbf n}
}.
\]

A negative square-root radicand is total internal reflection and is not repaired
numerically by taking an absolute value.

## 5. Three-dimensional polarization transport

The incident complex electric field is first projected transverse to its ray:

\[
\mathbf E_\perp=
\left(I-\hat{\mathbf s}_1\hat{\mathbf s}_1^T\right)\mathbf E.
\]

For non-normal incidence,

\[
\hat{\mathbf s}=\frac{\hat{\mathbf n}\times\hat{\mathbf s}_1}
{\|\hat{\mathbf n}\times\hat{\mathbf s}_1\|},
\]

\[
\hat{\mathbf p}_1=\hat{\mathbf s}_1\times\hat{\mathbf s},
\qquad
\hat{\mathbf p}_2=\hat{\mathbf s}_2\times\hat{\mathbf s}.
\]

The local amplitudes are

\[
E_s=\mathbf E_\perp\cdot\hat{\mathbf s},
\qquad
E_p=\mathbf E_\perp\cdot\hat{\mathbf p}_1.
\]

For isotropic nonmagnetic media,

\[
t_s=\frac{2n_1\cos\theta_i}
{n_1\cos\theta_i+n_2\cos\theta_t},
\]

\[
t_p=\frac{2n_1\cos\theta_i}
{n_2\cos\theta_i+n_1\cos\theta_t}.
\]

The transmitted vector is therefore

\[
\boxed{
\mathbf E_2=t_sE_s\hat{\mathbf s}+t_pE_p\hat{\mathbf p}_2
}.
\]

This operation is repeated at the flat and conical surfaces.  Because the cone
normal varies with azimuth, the local s/p basis and Fresnel weighting vary over a
tilted vector beam.  A single global 2x2 Jones matrix is therefore not used.

At exact normal incidence the plane of incidence is degenerate; the code uses the
polarization-independent normal-incidence Fresnel amplitude.

## 6. Interface energy gate

For each local mixed polarization state, the transmitted power ratio is

\[
T=\frac{n_2\cos\theta_t\,|\mathbf E_2|^2}
{n_1\cos\theta_i\,|\mathbf E_1|^2}.
\]

The reflected ratio is computed from \(r_s,r_p\).  For lossless real indices the
mandatory numerical identity is

\[
\boxed{R+T=1}.
\]

This is checked independently at both surfaces.

## 7. Finite optical path and eikonal phase

For a ray whose glass distance is \(L_g\) and whose downstream distance to the
chosen reference plane is \(L_e\),

\[
\mathrm{OPL}=n_gL_g+n_eL_e.
\]

The propagated phase relative to a common piston is

\[
\Phi=k_0\left(\mathrm{OPL}-\mathrm{OPL}_{\rm ref}\right),
\qquad
k_0=\frac{2\pi}{\lambda_0}.
\]

The incident complex vector phase is retained; the eikonal phase is multiplied
onto it rather than replacing it.

## 8. Fixed laboratory reference plane

Unlike the scalar eikonal reference, the calibrated vector bench route uses a
fixed laboratory plane

\[
z=z_{\rm ref}
\]

placed just downstream of all valid exit points.  For each outgoing ray,

\[
L_e=\frac{z_{\rm ref}-z_{\rm exit}}{s_{z,\rm out}},
\]

and

\[
x_{\rm ref}=x_{\rm exit}+L_es_{x,\rm out},
\qquad
y_{\rm ref}=y_{\rm exit}+L_es_{y,\rm out}.
\]

This intentionally preserves beam steering and decentre.  The field is never
recentred row-by-row or made to follow the tilted beam axis.

## 9. Ray-tube/Poynting amplitude transport

The mapping from physical flat entrance coordinates to the laboratory reference
plane has Jacobian

\[
J=\det
\begin{bmatrix}
\partial x_{\rm ref}/\partial x' & \partial x_{\rm ref}/\partial y'\\
\partial y_{\rm ref}/\partial x' & \partial y_{\rm ref}/\partial y'
\end{bmatrix}.
\]

A sign change within the valid footprint indicates a fold/caustic before the
reference plane and is rejected rather than hidden by interpolation.

For external medium index equal on input and output, normal Poynting-flux
conservation gives

\[
|\mathbf E_{\rm ref}|^2
(s_{z,\rm out})|J|
=
|\mathbf E_{\rm in}|^2
(\hat{\mathbf s}_{\rm in}\cdot\hat{\mathbf n}_{\rm ent})
T_1T_2.
\]

The complex Fresnel-transported polarization vector is normalized only by a real
magnitude, retaining all relative component phases, and this equation sets its
ray-tube amplitude.

## 10. Scalable inverse field remapping

The traced map is irregular on the output plane.  A production grid cannot use a
multi-million-point Delaunay triangulation, so Phase 2H inverts the smooth map:

1. fit a sparse affine map as an initial inverse estimate;
2. solve \((x',y')\mapsto(x_{\rm ref},y_{\rm ref})\) by Newton iteration;
3. interpolate the three complex laboratory components at the converged inverse
   coordinates;
4. reject pixels outside the valid ray footprint or above the inverse residual
   tolerance.

The local Jacobian amplitude already carries the geometrical ray density.

## 11. Maxwell transverse projection and spectral flux closure

Interpolation can introduce a small non-transverse component.  The regular field
is therefore projected in angular-spectrum space using

\[
P(\mathbf k)=I-\frac{\mathbf k\mathbf k^T}{|\mathbf k|^2}.
\]

For a projected field in a lossless dielectric, plane-integrated +z Poynting
flux is proportional to

\[
\int\!\!\int
n\frac{k_z}{k}
\left(|\tilde E_x|^2+|\tilde E_y|^2+|\tilde E_z|^2\right)
\,dk_xdk_y.
\]

The code evaluates the corresponding discrete Parseval sum and applies one final
*global* amplitude correction equal to the difference between expected traced
transmitted flux and the interpolated/projected field flux.  Local structure is
set by the ray-tube Jacobian; the global factor closes only interpolation loss.
Both the pre-correction factor and final closure ratio are recorded.

## 12. Sampling gate

A physically valid axicon can still be numerically unrepresentable.  The traced
outgoing directions require transverse spatial frequencies

\[
f_x=\frac{n_e}{\lambda_0}s_x,
\qquad
f_y=\frac{n_e}{\lambda_0}s_y.
\]

For grid spacing \(\Delta x\),

\[
f_{\rm Nyq}=\frac{1}{2\Delta x}.
\]

If the traced wavevectors approach/exceed the declared fraction of Nyquist, the
solver stops and demands a finer/smaller computational window.  This is
particularly important if a laboratory ``20 degree`` axicon specification turns
out to mean a large physical base angle.

## 13. Mandatory validation hierarchy

Before report figures can be authorised, Phase 2H requires:

1. zero-tilt recovery of the existing exact-Snell cone direction;
2. lossless Fresnel \(R+T=1\) at both surfaces;
3. finite-OPL gradient agreement with the traced outgoing transverse wavevector;
4. x/y rotational covariance for axisymmetric geometry with circular input;
5. positive/negative tilt mirror consistency in scalar ray metrics;
6. ray-tube flux closure before interpolation;
7. spectral normal-flux closure after interpolation/projection;
8. spectral transversality after projection;
9. hard rejection of under-sampled large-cone cases;
10. preservation of all existing scalar refractive and Phase 2G vector/objective
    regression gates.

Only after these pass is the calibrated segmented-vector bench route permitted to
replace its previous rigid-tilt guard with the Phase 2H solver.
