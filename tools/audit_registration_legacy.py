#!/usr/bin/env python3
"""Reproduce the historical propagation failure without changing accepted evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from vbb_study.digital_twin.slm_registration_architectures import (
    build_architecture_registration_route,RegistrationState,RegistrationSampling)
from vbb_study.digital_twin.slm_registration_metrics import complex_fidelity

def run(out):
    rows=[]
    for w in [50,250]:
        for pixel in ['area_average','centre_sample']:
            fields=[];powers=[]
            for d in [0,.5]:
                r=build_architecture_registration_route('V20',architecture='upstream_vortex',
                    registration=RegistrationState(slm1_dx_m=d*8e-6),
                    sampling=RegistrationSampling(2500,768,.01),beam_radius_m=w*8e-6,pixel_value_model=pixel)
                E=r['field_on_axicon_plane'];fields.append(E);powers.append(float(np.sum(abs(E)**2)))
                m=r['metadata']
                rows.append(dict(w_px=w,pixel_model=pixel,offset_px=d,
                    propagation=m['numerical_propagation_audit'],
                    quantitative_registration_claims_allowed=m['quantitative_registration_claims_allowed'],
                    decimation=m['decimation'],iris=m['fourf']['iris_selected_power_fraction']))
            rows[-1].update(raw_infidelity=1-complex_fidelity(fields[1],fields[0]),pre_power_ratio=powers[1]/powers[0])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=Path('outputs/validation/registration_definitive/legacy_audit.json'))
    run(a.parse_args().out)
