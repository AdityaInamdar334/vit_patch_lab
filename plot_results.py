"""Export publication-ready figures from the saved measured JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from vit_lab.data import SpatialMarkers

root=Path(__file__).resolve().parent
result=json.loads((root/'results/benchmark.json').read_text())
summary=result['summary']
order=['cnn','vit8','vit4','vit8_no_pos','vit8_mean']
labels=['CNN','ViT / 8px','ViT / 4px','ViT / 8px no position','ViT / 8px mean pool']
positions=np.arange(len(order))
fig,ax=plt.subplots(figsize=(11,5.4))
for offset,metric,label in ((-.27,'clean_accuracy','Clean'),(0,'noisy_accuracy','High noise'),(.27,'shifted_accuracy','Translated markers')):
    values=[100*summary[name][metric]['mean'] for name in order]
    ax.bar(positions+offset,values,width=.26,label=label)
ax.axhline(50,linestyle=':',linewidth=1,color='gray')
ax.set_ylim(0,108)
ax.set_xticks(positions,labels,rotation=12,ha='right')
ax.set_ylabel('Held-out accuracy (%)')
ax.set_title('Inductive bias and robustness on a controlled two-marker task')
ax.legend(frameon=False)
fig.tight_layout()
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/01_clean_noise_translation.png',dpi=180)
plt.close(fig)

low=json.loads((root/'results/low_data.json').read_text())['summary']
fig,ax=plt.subplots(figsize=(8,4.5))
for name in ('cnn','vit4'):
    xs=[120,1200]
    ys=[100*low[name]['clean_accuracy']['mean'],100*summary[name]['clean_accuracy']['mean']]
    ax.plot(xs,ys,marker='o',label='CNN' if name=='cnn' else 'ViT / 4px')
ax.set_xticks([120,1200])
ax.set_xlim(0,1320)
ax.set_ylim(75,102)
ax.set_ylabel('Held-out accuracy (%)')
ax.set_xlabel('Training images available (fixed 150 gradient steps)')
ax.set_title('Data-volume ablation on the synthetic task')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root/'figures/02_training_data.png',dpi=180)
plt.close(fig)

sample=SpatialMarkers(2,seed=19)
fig,ax=plt.subplots(figsize=(5,5))
ax.imshow(sample.images[0].permute(1,2,0).numpy())
for i in range(0,33,8):
    ax.axhline(i-.5,color='white',linewidth=.6,alpha=.7)
    ax.axvline(i-.5,color='white',linewidth=.6,alpha=.7)
ax.set_xticks([]);ax.set_yticks([])
ax.set_title('A 32 x 32 sample divided into 16 patches of 8 x 8')
fig.tight_layout()
fig.savefig(root/'figures/03_patch_grid.png',dpi=180)
plt.close(fig)
print('Saved 3 figures in figures/')
