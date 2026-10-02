#!/usr/bin/env python3
"""Replace the first three summary pages with audited physical-scale evidence.

The remaining bench metrics / scaling / architecture pages are preserved.
"""
import argparse,io
from pathlib import Path
from PIL import Image
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

def build(source,figures,out):
    for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold')
    temp=out.with_name(out.stem+'_front.tmp.pdf');W,H=595.28,841.89;M=42;CW=W-2*M
    c=canvas.Canvas(str(temp),pagesize=(W,H));y=0
    body=ParagraphStyle('body',fontName='Body',fontSize=10.8,leading=15.8,textColor=HexColor('#233044'))
    cap=ParagraphStyle('cap',parent=body,fontSize=9,leading=13,textColor=HexColor('#526174'))
    def para(text,caption=False):
        nonlocal y
        p=Paragraph(text,cap if caption else body);_,h=p.wrap(CW,H)
        assert y-h>62,(y,h,text[:60])
        p.drawOn(c,M,y-h);y-=h+10
    def head(text):
        nonlocal y
        c.setFillColor(HexColor('#166579'));c.setFont('Bold',13);c.drawString(M,y-13,text);y-=28
    def image(name,maxheight):
        nonlocal y
        im=Image.open(figures/name).convert('RGB');im.thumbnail((2000,2000),Image.Resampling.LANCZOS)
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=95,subsampling=0)
        iw,ih=im.size;sc=min(CW/iw,maxheight/ih);dw,dh=iw*sc,ih*sc
        c.drawImage(ImageReader(buf),M+(CW-dw)/2,y-dh,width=dw,height=dh);y-=dh+12
    def page(n,title,sub):
        nonlocal y
        if n>1:c.showPage()
        c.setFillColor(HexColor('#166579'));c.rect(0,H-10,W,10,stroke=0,fill=1)
        c.setFont('Bold',9);c.drawString(M,H-36,'SLM PIXEL REGISTRATION  /  MASTER SUMMARY')
        c.setFillColor(HexColor('#172538'));c.setFont('Bold',22);c.drawString(M,H-72,title)
        c.setFillColor(HexColor('#526174'));c.setFont('Body',10);c.drawString(M,H-94,sub)
        c.setStrokeColor(HexColor('#D7E0E6'));c.line(M,43,W-M,43)
        c.setFont('Body',8);c.drawString(M,29,'Physical-scale revision | 2 October 2026 | Input radius is not Bessel ring radius')
        c.drawRightString(W-M,29,str(n));y=H-117
    page(1,'The small beam really is smaller','First check the footprint on the SLM, before looking at a downstream ring')
    para('<b>50 pixels means a 0.4 mm radius on the SLM. 250 pixels means a 2 mm radius.</b> These are genuinely different incident beams: the small one is five times narrower. The earlier summary jumped straight to a central ring after the axicon, which obscured this distinction.')
    image('01_input_radius_verified.png',260)
    para('<b>Figure 1.</b> Saved input fields on identical millimetre axes. Cyan circles mark the Gaussian 1/e² intensity radii. The cross-sections use the same x axis and show the fivefold width difference. Each image is normalised to its own peak; these panels do not compare total laser power.',True)
    head('The width was measured from the saved fields')
    para('A Gaussian fit recovers <b>0.400 mm</b> and <b>2.000 mm</b>. Their intensity FWHM diameters are <b>0.471 mm</b> and <b>2.355 mm</b>. The 50-pixel beam was not replaced by, or resized into, the 250-pixel beam. Adding a phase-only vortex leaves the intensity unchanged at the SLM surface.')
    head('How would we make the small beam in the laboratory?')
    para('Use a smaller incident Gaussian beam, for example through a beam-reducing telescope. Going from a 2 mm to a 0.4 mm radius requires a <b>fivefold beam reduction</b>. In the simulation this is set directly in the incident amplitude. A pupil that clips the beam would be a different test, not the same Gaussian beam reduction.')
    para('<b>A smaller final Bessel ring is a separate objective.</b> Its size is controlled mainly by vortex charge and the axicon cone angle. Lower charge, a larger cone angle, or suitable downstream demagnification can reduce that ring. Reducing the incident footprint alone does not guarantee a smaller central ring.')
    page(2,'What changes before the axicon?','The same two input beams, propagated through the two-SLM route')
    para('Both rows use L = 20, Architecture A, blaze on both SLMs and the conditional 200 mm free-space gap. The vortex-plus-blaze panel is shifted by half a pixel while the nominal beam and hologram centres remain together.')
    image('02_before_axicon.png',430)
    para('<b>Figure 2.</b> Selected field immediately before the physical axicon. Left: zero registration. Middle: half a pixel in x. Right: shifted minus baseline intensity. Both rows have the same physical crop and residual range. Row labels state the <i>input</i> radius on SLM1; they are not radii measured at this plane.',True)
    head('A small input can spread into a larger ring')
    para('A vortex imposed on a small Gaussian beam has a broader angular spectrum and diffracts more during the 200 mm transfer. The pre-axicon ring is therefore not simply a fivefold smaller copy of the large-beam ring. A radius specified at SLM1 does not stay fixed throughout the optical system.')
    para('The half-pixel intensity-shape changes are <b>8.82% for the 0.4 mm input</b> and <b>0.805% for the 2 mm input</b>. This is the norm of the difference between power-normalised images relative to the baseline image; it is not a percentage of power lost.')
    page(3,'Why are the final rings similar?','The central Bessel ring is set by charge and cone angle, not the input radius')
    para('These images show <b>12 mm downstream of the axicon</b>, on micrometre axes. They are a 300 micrometre central crop, not pictures of the input beam footprint. Keeping a common physical crop makes the registration differences directly comparable.')
    image('03_downstream_ring_crop.png',425)
    para('<b>Figure 3.</b> The same two input radii as Figures 1 and 2. Both rows use the same downstream distance, crop and signed-difference range. Each row is normalised to its own baseline peak. Similar central ring radii do not mean identical envelopes, absolute brightness or axial behaviour.',True)
    head('An independent check of the expected ring scale')
    para('For an ideal vortex Bessel field, intensity varies as <b>J<sub>L</sub>(k<sub>r</sub>r)²</b>. The first bright ring occurs where this Bessel function has its first positive maximum. With L = 20 and the model\'s cone wavevector, this gives approximately <b>21 micrometres</b>. The Gaussian input radius does not appear in that ideal ring-scale relation. Finite-beam diffraction can still perturb it.')
    para('The two cached fields were propagated independently using exact scalar angular-spectrum propagation at approximately <b>0.992 micrometres per sample</b>. No resizing was applied to force the rings to match. Scalable ASM is useful when numerical windows must change, but it should not change the physical answer. At this plane, the half-pixel shape changes remain <b>5.45%</b> and <b>0.294%</b> for the small and bench input beams respectively.')
    c.save();old=PdfReader(source);assert len(old.pages)==6
    writer=PdfWriter()
    for p in PdfReader(temp).pages:writer.add_page(p)
    for p in old.pages[3:]:writer.add_page(p)
    writer.add_metadata({'/Title':'SLM registration: plain-English summary, physical-scale revision','/Author':'Structured-beam PhD study'})
    with out.open('wb') as f:writer.write(f)
    temp.unlink();print(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();build(a.source,a.figures,a.out)
