# LAB README - Downstream Carrier/Stop Session

Run: <run_id>
Timestamp: <timestamp_utc>
Git commit: <git_commit>

## Beam state

- SLM1: flat phase
- SLM2: command-domain carrier-only mask
- camera: installed downstream final-focus/output plane
- Fourier stop: state recorded and varied deliberately where possible
- axicon: state recorded; bypass/remove only if physically practical
- all downstream optics: recorded as fixed bench state

## Minimum first session

1. Record run ID, camera position, SLM identifiers, stop state,
   axicon state, and every fixed downstream optic.
2. Capture dark frames at the intended exposure/gain.
3. Display SLM1-flat / SLM2-flat and capture a flat-reference output.
4. Keep the stop at one recorded baseline setting.
   Run the x-carrier sequence and capture one raw image per mask.
5. Keep the same baseline stop setting.
   Run the y-carrier sequence and capture one raw image per mask.
6. Pick one carrier setting that visibly produces a usable output.
   Perform a small stop-centre x/y sweep if adjustable.
7. If the stop aperture/radius is adjustable, perform a small stop-size sweep.
8. Repeat the apparent best configuration at least three times.
9. Do not move camera, lenses, SLMs, or downstream optics without logging it.
10. Keep raw camera files unchanged under the generated run ID.

## Explicit statement

This session measures the downstream optical response of the complete existing route.

It does not directly image the Fourier plane.

It does not establish physical Fourier-plane coordinates, selected-order purity,
or a fully calibrated physical 4F model.

Warning: With the installed downstream camera, this experiment characterises the final response to carrier and stop settings. It does not directly measure Fourier-plane order positions or by itself calibrate the physical 4F coordinate system.
