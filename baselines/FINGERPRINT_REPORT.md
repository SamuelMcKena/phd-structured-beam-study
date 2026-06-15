# Fingerprint Report

No OUT OF RANGE fingerprints were found.

This report is a human-facing sanity cross-check only. Pytest assertions compare the committed exact payloads, not these approximate fingerprints.

Array hashing: `sha256(np.ascontiguousarray(a).tobytes())`.
Endianness assumption: Hashes use native-endian NumPy bytes from this platform; dtype_str records byte order and ENVIRONMENT.json records sys.byteorder.

| Fingerprint | Target | Status | Match |
| --- | --- | --- | --- |
| Objective demagnification | 0.008071 +/- 1% | IN RANGE | `{"case_id": "general_holographic_ideal", "key": "metrics.magnification_to_sample", "value": 0.008071063517253126}` |
| Physical-axicon k_r | 1.603e6 m^-1 +/- 2% | IN RANGE | `{"case_id": "general_physical_ideal", "key": "axicon_metadata.k_r_m_inv", "value": 1603333.3333333333}` |
| SLM2 residual phase RMS | starts about 1.81 rad, flattens about 0.007 rad (report window: before +/-5%, after +/-25% or 0.002 rad) | IN RANGE | `{"after": 0.00712673122363286, "before": 1.8138168507664694, "case_id": "general_physical_lab", "key": "axicon_metadata.slm2_residual_phase_rms_before_rad/after_rad"}` |
| Holographic first-order selected fraction | 0.73 +/- 0.03 | IN RANGE | `{"case_id": "general_holographic_lab", "key": "metrics.first_order_selected_fraction", "value": 0.732111295550108}` |
| Bessel zone by route | holographic about 35 um, physical about 114 um (report window: +/-15%) | IN RANGE | `{"holographic": {"case_id": "general_holographic_lab", "key": "metrics.canonical_zone_um", "route": "holographic", "value": 38.01556727371095}, "physical": {"case_id": "general_physical_ideal", "key": "metrics.canonical_zone_um", "route": "physical", "value": 113.94754279265943}}` |
