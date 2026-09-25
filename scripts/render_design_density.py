"""Render verified policy density in the author's original four viewports."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize,LinearSegmentedColormap
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]/'data/design_density'
def project(p):
    p=np.asarray(p) @ np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
    direction=np.array([3.5,-5.5,6.]);direction/=np.linalg.norm(direction)
    right=np.cross(-direction,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,-direction)
    return p @ np.stack([right,up,direction],axis=1)
CMAP=LinearSegmentedColormap.from_list('density',[(0,'#f5f8f8'),(.2,'#b8dada'),(.5,'#5ca9b1'),(1,'#174e58')])
NORM=Normalize(0,5)
TARGETS=[('image86','airfoil2d','initial'),('image87','airfoil2d','final'),('image98','superwing','initial'),('image99','superwing','final')]

def main():
 out=ROOT/'panels';out.mkdir(exist_ok=True);records=[]
 for media,case,stage in TARGETS:
  z=np.load(ROOT/f'data/{case}_{stage}.npz');W,H={'image86':(1124,444),'image87':(1124,444),'image98':(638,327),'image99':(821,374)}[media]
  if case=='airfoil2d':
   lims=[(-.16,1.16),(-.26,.26)];points=z['coordinates'].reshape(-1,2)[z['indices']]
  else:
   q=project(z['faces']);lo=q[:,:,:2].min((0,1));hi=q[:,:,:2].max((0,1));lims=list(zip(lo,hi))
   signed=np.cross(q[:,1]-q[:,0],q[:,3]-q[:,0])[:,2]
   ids=z['indices'][signed[z['indices']]<0];points=project(z['positions'][ids])[:,:2]
  for mode in ['density','samples','combined']:
   fig=plt.figure(figsize=(W/100,H/100),dpi=200,facecolor='none');ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
   ax.set_xlim(*lims[0]);ax.set_ylim(*lims[1])
   if mode!='samples':
    if case=='airfoil2d':
     xy=z['coordinates'];ax.pcolormesh(xy[:,:,0],xy[:,:,1],z['density'].reshape(221,51),cmap=CMAP,norm=NORM,shading='gouraud',rasterized=True)
     ax.fill(*z['outline'].T,facecolor='white',edgecolor='#596970',lw=.35,zorder=3)
    else:
     order=np.argsort(q[:,:,2].mean(1))
     ax.add_collection(PolyCollection(q[order,:,:2],facecolors=CMAP(NORM(z['density'][order])),edgecolors='none',lw=0,antialiased=False,rasterized=True))
   if mode!='density':
    radius=1.15 if case=='airfoil2d' else .56
    ax.scatter(points[:,0],points[:,1],s=(radius*4*72/200)**2,c='#183d48',alpha=.85,edgecolors='white',linewidths=.05 if case=='airfoil2d' else 0,zorder=5)
   prefix=out/f'{case}_{stage}_{mode}'
   fig.savefig(prefix.with_suffix('.png'),dpi=200,transparent=True,pad_inches=0)
   if mode!='density':
    fig.savefig(prefix.with_suffix('.pdf'),dpi=200,transparent=True,pad_inches=0)
    fig.savefig(prefix.with_suffix('.svg'),dpi=200,transparent=True,pad_inches=0)
   plt.close(fig)
  inside=(points[:,0]>=lims[0][0])&(points[:,0]<=lims[0][1])&(points[:,1]>=lims[1][0])&(points[:,1]<=lims[1][1])
  records.append(dict(media=media,case=case,stage=stage,viewport=lims,selected_count=len(z['indices']),displayed_count=int(inside.sum()),native_density_min=float(z['density'].min()),native_density_max=float(z['density'].max()),colorbar_range=[0,5],samples_layer='actual saved indices; back-facing wing samples hidden'))
 # Independent common legend, same gradient as the preserved author colorbar.
 (ROOT/'legends').mkdir(exist_ok=True)
 Image.fromarray((CMAP(np.tile(np.linspace(0,1,1600),(30,1)))*255).astype('uint8')).save(ROOT/'legends/gradient.png')
 fig,ax=plt.subplots(figsize=(4,.65));fig.subplots_adjust(left=.03,right=.97,top=.94,bottom=.60)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=ax,orientation='horizontal');cb.set_ticks([0,1,2.5,5]);cb.set_label('Density (uniform = 1)',fontsize=9);cb.ax.tick_params(labelsize=9,length=2,pad=1);cb.outline.set_linewidth(.4)
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'legends/density.{ext}',dpi=300,transparent=True)
 plt.close(fig)
 (ROOT/'render_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 # User-review contact sheet.
 canvas=Image.new('RGB',(1600,880),'white')
 from PIL import ImageDraw
 draw=ImageDraw.Draw(canvas)
 for i,(_,case,stage) in enumerate(TARGETS):
  im=Image.open(out/f'{case}_{stage}_combined.png');im.thumbnail((760,345))
  x=20+800*(i%2);y=45+440*(i//2);canvas.paste(im,(x,y),im)
  draw.text((x,y-24),f'{case} {stage} | actual policy density + saved samples',fill='#172c33')
 canvas.save(ROOT/'density_preview.png')
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
