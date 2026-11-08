#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



benign_mutants = [
    'WT',
    'V904I',
    'I926N',
    'I966V',
    'F712L',
    'S720F',
    'P741S',
    'I744M',
    'V774L',
    'R831H'
]


# Read data
df = pd.read_csv("plot-relative-population-active-multi-traj.csv")
print('Reading CSV file', flush=True)



# Extract mutant name
df["Mutant"] = (df["Mutants"] .str.replace("egfr-", "", regex=False) .str.replace("-active-charmm-1", "", regex=False).str.replace("-active-charmm-2", "", regex=False) .str.replace("-active-charmm-3", "", regex=False))



replacements = {
    "delE746-A750": r"$\Delta$E746-A750",
    "delL747-E749,A750P": r"$\Delta$L747-E749,A750P",
    "delL747delS752,P753S": r"$\Delta$L747,$\Delta$S752,P753S",
    "delL747-S752,P753S": r"$\Delta$L747-S752,P753S",
    "delE746-A750, T790M":r"$\Delta$E746-A750,T790M"
}

df["Mutant"] = df["Mutant"].replace(replacements)



# Calculate statistics
print('Calculating statistics', flush=True)
stats = ( df.groupby("Mutant")["Active"] .agg(['mean','std', 'count']) .reset_index())


stats["SE"] = stats["std"] / np.sqrt(stats["count"])
# Sort by mean
stats = stats.sort_values("mean").reset_index(drop=True)

order = stats["Mutant"].tolist()
#df["Mutant"] = pd.Categorical( df["Mutant"], categories=order, ordered=True)


# plot individual replicas
fig, ax = plt.subplots(figsize=(10, 14))
y = np.arange(len(stats))

#for i, mut in enumerate(stats["Mutant"]):
#	vals = df[df["Mutant"] == mut]["Active"].values
	#jitter so points don't overlap
#	jitter = np.linspace(-0.12, 0.12, len(vals))
#	if mut=='wt':
#		ax.scatter(vals,np.full(len(vals),i)+jitter,s=100,marker='s',alpha=0.8,zorder=5,label="WT replicas")
#	else:
#		ax.scatter(vals,np.full(len(vals),i)+jitter,s=50,alpha=0.8,zorder=3)

    

# plot mean ± SE
ax.errorbar(stats["mean"], y, xerr=stats["SE"],fmt='D', markersize=8, capsize=5, linewidth=2,color='black', zorder=3)

wt_row = stats[stats["Mutant"].str.upper() == "WT"]
print(wt_row)

if not wt_row.empty:
	wt_y =stats.index[stats["Mutant"].str.upper() == "WT"][0]
	ax.errorbar(wt_row["mean"].values[0], wt_y, xerr=wt_row["SE"].values[0],fmt='s', markersize=10, capsize=7, linewidth=2,color='black', zorder=5)
#	ax.axvline(wt_row["mean"].values[0], linestyle='--',linewidth=1.2,color='black', alpha=0.5)
ax.set_yticks(y) 
ax.set_yticklabels(stats["Mutant"])

for label in ax.get_yticklabels():
  mutation = label.get_text().upper()
  
  if mutation in benign_mutants:
    label.set_fontweight('bold')
  else:
    label.set_fontweight('normal')

ax.set_xlabel("Fraction Active", fontsize=14)
ax.set_ylabel("Mutant", fontsize=14)
ax.tick_params(axis='y',labelsize=10)
ax.tick_params(axis='x',labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(left=0.30)
plt.savefig("Fraction_Active_Mean_SE-2.png", dpi=600,bbox_inches='tight')
plt.show()
print('Done')





