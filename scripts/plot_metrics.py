# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import json
import seaborn as sns

from matplotlib.patches import Patch

trial_colors = {
    "trial1": "#b00303",
    "trial2": "#000000",
    "trial3": "#000000",
    "trial4": "#b00303",
    "trial5": "#000000",
    "trial6": "#b00303",
    "trial7": "#000000",
    "trial8": "#b00303",
    "trial9": "#000000",
    "trial10": "#b00303",
    "trial11": "#000000",
    "trial12": "#b00303",
}


pretty_names = {'C1':r"$\text{DAC}_{(29)}$",
                'C2':r"$\text{HE-NAC}_{(24, 2)}$",
                'C3':r"$\text{HE-NAC}_{(18, 2, 1)}$",
                'C4':r"$\text{HE-AAC}$",
                'reference':r"Hidden Ref.",
                'anchor35':r"Low Anchor"}

def plotBoxMetrics(results, plotTitle=''):

	font = {'family' : 'normal',
		'size'   : 30}
	matplotlib.rc('font', **font)

	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	matplotlib.rcParams['axes.unicode_minus'] = False

	f = plt.figure(figsize=(10,11))
	names = ['reference','C4','C3','C2','C1','anchor35']
	xs     = []
	vals   = []
	colors = []
	for i, name in enumerate(names):
		val, col = [o[0] for o in results[name]], [o[1] for o in results[name]]
		vals.append(np.asarray(val,dtype=int))
		colors.append(np.asarray(col))
		xs.append(np.random.normal(i + 1, 0.04, len(results[name])))

	#convert names
	real_names = [pretty_names[i] for i in names]

	sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
	boxprops = dict(linestyle='-', linewidth=2.5, color='#00145A', facecolor='white')
	flierprops = dict(marker='o', markersize=0,
					  linestyle='none')
	whiskerprops = dict(color='#00145A')
	capprops = dict(color='#00145A')
	medianprops = dict(linewidth=2.0, linestyle='-', color='#6e0101')
	meanprops = dict(linewidth=2.0, linestyle='--', color='#017507')
	bb = plt.boxplot(vals, labels=real_names,
		notch=True, 
		boxprops=boxprops, 
		whiskerprops=whiskerprops,
		capprops=capprops, 
		flierprops=flierprops, 
		medianprops=medianprops,
		showmeans=True,
		meanline=True,
		meanprops=meanprops,
		showfliers=True,
  		patch_artist=True, 
    	zorder=1) 


	p1 = np.array(sns.color_palette("Greens_r", 10))[2:5,...]
	p2 = np.array(sns.color_palette("Blues_r", 10))[2:5,...]
	palette = np.concatenate([p1,p2])
	alphas = (np.ones(palette.shape[0])*0.7).reshape(-1,1)
	palette = np.append(palette,alphas,axis=1)
	# for patch, color in zip(bb['boxes'][2:-1], palette):
	# 	patch.set_facecolor(color)
	# 	patch.get_text()
 
	for i, (x, val, col) in enumerate(zip(xs, vals, colors)):
		low = np.quantile(val,0.25)
		high  = np.quantile(val,0.75)
		val_n = val[(val > low) & (val < high)]
		x_n = x[(val > low) & (val < high)]
		col_n = col[(val > low) & (val < high)]
		plt.scatter(x_n, val_n, alpha=0.7, color='black', zorder=10000)

	# # Manual legend
	# legend_elements = [
	# 	Patch(facecolor='#b00303', edgecolor='#b00303', label='HB-Prominent'),
	# 	Patch(facecolor='#000000', edgecolor='#000000', label='Random')
	# ]

	# plt.legend(handles=legend_elements)
	plt.ylabel('Subjective Score')
	plt.xticks(np.arange(len(real_names))+1, real_names, rotation=90)
	plt.tight_layout()
	plt.savefig('/N/slate/daripete/jstsp-dac/plots/mushra.png')
	plt.savefig('/N/slate/daripete/jstsp-dac/plots/mushra.pdf')


def plot_mushra():
	stimuli = ['C1','C2','C3','C4','reference','anchor35']
	results = {i:[] for i in stimuli}
	path = './mushra.json'
	with open(path) as json_file:
		data = json.load(json_file)

	participant_ids = list(data.keys())
	for p_id in participant_ids:
		all_trials = json.loads(data[p_id])['trials']
		for trials in all_trials: 
			for trial in trials['responses']:
					results[trial['stimulus']].append((trial['score'], trial_colors[trials['id']]))
	plotBoxMetrics(results)

plot_mushra()
