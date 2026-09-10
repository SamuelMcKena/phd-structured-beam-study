# Surface mapping

Zemax surface numbers are never assumed to equal the repository's optical stage names. Use an explicit JSON mapping derived from the inspected prescription.

```json
{
  "input_plane": 1,
  "after_slm1": null,
  "fourier_plane": null,
  "spatial_filter": null,
  "after_slm2": null,
  "after_axicon": null,
  "objective_entrance": null,
  "sample_surface": null
}
```

Every non-null surface index is validated against the prescription before POP runs. The current Phase 3 runner requires an input (`objective_entrance` preferred, otherwise `input_plane`) and `sample_surface` for a reportable POP propagation. Surface index `0` is valid and is never discarded by truth-value shortcuts.
