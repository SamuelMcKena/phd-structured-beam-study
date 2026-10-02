#!/usr/bin/env python3
"""Package complete ignored evidence into two portable, integrity-checked parts."""
import argparse, hashlib, json, subprocess, zipfile
from pathlib import Path

def run(data,out):
    required=['metrics.csv','convergence.csv','post_convergence.csv','additional_diagnostics.csv',
              'manifest_sweep.json','manifest_controls.json','manifest_postcontrols.json','manifest_extra.json',
              'local_reference/sweep.csv','local_reference/unit_cell.csv','local_reference/scaling_fit.json']
    missing=[p for p in required if not (data/p).is_file()]
    if missing:raise RuntimeError('Incomplete study: '+', '.join(missing))
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    files=sorted(p for p in data.rglob('*') if p.is_file())
    entries=[]
    for p in files:
        supplemental=p.suffix=='.npz' and p.name.startswith(('control_','actual_panel_','axial_','immediate_slm1_'))
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        entries.append(dict(path=str(p.relative_to(data)),part=2 if supplemental else 1,
            size_bytes=p.stat().st_size,sha256=digest))
    inventory=json.dumps(dict(packaging_source_commit=commit,files=entries),indent=2)
    readme=f'''SLM PIXEL REGISTRATION WITH APPROXIMATE 200 MM INTER-PANEL FREE SPACE

Source: https://github.com/SamuelMcKena/phd-structured-beam-study/pull/15
Branch: slm-registration-architecture-comparison
Packaging source commit: {commit}

Part 1, SLM_Registration_200mm_Evidence.zip: every CSV and assumption manifest,
the full five-offset complex-field sequences, refined post-axicon fields,
the complete local charge/radius/unit-cell layer, and all standalone figures.
Part 2, SLM_Registration_200mm_Controls.zip: full convergence coefficient
arrays, actual SLM2-plane fields and common-z axial complex fields.
Extract both parts into the SAME directory. They do not overwrite data files.
The identical file_inventory.json in each part gives SHA256, byte size and part
for every numerical/figure file. Reports are supplied separately as DOCX/PDF.

Both SLMs carry an 8 um lattice and nominal 20-pixel +x blaze. This is the
FLAT-CORRECTION PHYSICS-ISOLATION BASELINE. The 200 mm gap is approximate and
user-reported. Aligned coordinates, same-sign carriers, ideal unit-magnification
relay and ideal 20-degree BASE-angle axicon are CONDITIONAL assumptions.
The relay and physical iris follow BOTH SLMs. No inter-panel iris is inserted.
Numerical bandwidth is not a physical iris. The selected iris adapts to L,w,
then stays fixed across registration offsets and architectures within a case.

The earlier identity-transfer evidence remains historical and separate.
Its exact A/B role equality is not a theorem for finite inter-panel distance.
Do not claim an experimentally superior architecture from this baseline.
Do not relabel the 25/50-pixel stress radii as the present 250-pixel beam.
No immediate intensity change is assigned to the phase-only axicon.

REPRODUCE FROM A CLEAN CHECKOUT OF THE SOURCE COMMIT
Python 3.12; numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 matplotlib==3.10.8
python-docx==1.2.0; pytest; Pandoc and LibreOffice for document export.
Set PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -m pytest -q tests/test_slm_pixel_registration.py tests/test_slm_registration_architectures.py tests/test_registration_reference.py tests/test_registration_interpanel.py
python -m tools.run_registration_interpanel_study --stage local
python -m tools.run_registration_interpanel_study --stage sweep
python -m tools.run_registration_interpanel_study --stage controls
python -m tools.run_registration_interpanel_study --stage postcontrols
python -m tools.run_registration_interpanel_study --stage extra
python -m tools.refine_registration_interpanel_figures --data outputs/validation/registration_interpanel_200mm
python -m tools.plot_registration_interpanel_study --data outputs/validation/registration_interpanel_200mm --output outputs/validation/registration_interpanel_200mm/figures
python -m tools.build_registration_interpanel_report --data outputs/validation/registration_interpanel_200mm --output outputs/reports/registration_interpanel
python -m tools.package_registration_interpanel_evidence --data outputs/validation/registration_interpanel_200mm --output outputs/delivery/registration_interpanel

Alternatively extract both evidence parts and regenerate plots/report without
repeating simulations, using their extracted directory as --data.
Native editable equations are preserved in DOCX. Every delivered report page
was rendered and visually reviewed; CI PDF export alone is not visual QA.
Delivered propagated XY sequences use 1 um sampling; primary 2 um sequences
and endpoint convergence controls remain preserved separately. Shape
residuals use a fixed 300 um-wide XY region after propagation. Field fidelity
uses the full selected spectrum. Those two metrics are not interchangeable.
The manifests describe the calculation source; the inventory identifies the
packaging source. Large evidence is intentionally not tracked in Git.
'''
    (out/'Reproduction_README.txt').write_text(readme)
    (out/'file_inventory.json').write_text(inventory)
    for part,name in [(1,'SLM_Registration_200mm_Evidence.zip'),(2,'SLM_Registration_200mm_Controls.zip')]:
        destination=out/name;temp=destination.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for entry in entries:
                if entry['part']==part:z.write(data/entry['path'],entry['path'])
            z.writestr('file_inventory.json',inventory);z.writestr('Reproduction_README.txt',readme)
        temp.replace(destination)
        print(destination,destination.stat().st_size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.output)
