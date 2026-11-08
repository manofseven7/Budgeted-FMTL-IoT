"""Exact vector diagram of the active experimental pipeline."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def draw(path):
    fig,ax=plt.subplots(figsize=(9,3.0));ax.set(xlim=(0,10),ylim=(0,3.4));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=color,edgecolor='#334155',linewidth=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#334155',linewidth=.9))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,ha='center',fontsize=6)
    box(.15,2.05,2.1,.85,'Local inputs and targets\nTraining-only normalization','#e0f2fe')
    box(2.85,2.05,2.1,.85,'Shared tanh representation\nTwo task heads','#dcfce7')
    box(5.55,2.05,1.85,.85,'Local parameter deltas\nClient/task residuals','#fef3c7')
    box(8.05,2.05,1.8,.85,'TopK + overlap rule\nValues, indices, framing','#ede9fe')
    box(.3,.3,2.25,.85,'Budget + participation deficit\nSelect clients; log frequencies','#e2e8f0')
    box(3.3,.3,3.0,.85,'Aggregate compatible task updates\nLog uplink, control and downlink','#f1f5f9')
    box(7.0,.3,2.75,.85,'Intel only: fixed current-value offset\nHeads learn next-value residuals','#ffedd5')
    arrow((2.25,2.47),(2.85,2.47));arrow((4.95,2.47),(5.55,2.47));arrow((7.4,2.47),(8.05,2.47))
    arrow((1.43,1.15),(1.43,2.05));arrow((8.95,2.05),(6.3,.8));arrow((3.3,.72),(2.55,.72))
    arrow((4.8,1.15),(3.9,2.05),'Model delivery');arrow((7.1,1.1),(4.6,2.05),'Common sensor model')
    fig.tight_layout(pad=.1);fig.savefig(path,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':draw(Path(sys.argv[1]) if len(sys.argv)>1 else Path('fig_architecture.pdf'))
