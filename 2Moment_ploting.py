import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import numpy as np

# Set the global font to be Arial, size 10 (or any other size you prefer)
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'

df = pd.read_csv('moment.csv',sep='\s+')
Ecce_pile = 19.2+min(df.iloc[:,1])
fig, ax =  plt.subplots(figsize=(6, 6))
ax.plot(-df[r'SM.SM1'],-df.iloc[:,1]+Ecce_pile, color='black')
plt.xlabel(r'Bending moment (kN·m)', fontname='Arial', labelpad=14, fontsize=12)
plt.ylabel(r'Depth (m) ', fontname='Arial', fontsize=12)

plt.gca().spines['left'].set_position(('data',0))
plt.gca().spines['bottom'].set_position(('data',0))
plt.gca().xaxis.set_label_position('top') # set the x-axis label position to top
plt.gca().spines['right'].set_visible(False) # hide the right spines
plt.gca().spines['top'].set_visible(False) # hide the top spines
plt.gca().tick_params(axis='x', direction='out', labelsize=10)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().tick_params(axis='y', direction='out', labelsize=10)
plt.ylim(19.2,0)
plt.savefig('moment.jpg',format='jpg',dpi=300)