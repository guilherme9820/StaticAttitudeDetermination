import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mean_sigma = pd.read_csv("mean_sigma.csv")

sns.set(style='ticks', context='talk')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['legend.title_fontsize'] = 24

groups = mean_sigma.groupby("algorithm")

plt.figure(figsize=(10, 8))

sns.regplot(logx=True, scatter=False, data=groups.get_group("nn"), x="sigma", y="loss", label="neural network")
sns.regplot(logx=True, scatter=False, data=groups.get_group("svd"), x="sigma", y="loss", label="svd")
sns.regplot(logx=True, scatter=False, data=groups.get_group("q_method"), x="sigma", y="loss", label="q-method")
sns.regplot(logx=True, scatter=False, data=groups.get_group("quest"), x="sigma", y="loss", label="quest")
chart = sns.regplot(logx=True, scatter=False, data=groups.get_group("esoq2"), x="sigma", y="loss", label="esoq2")

chart.set_yscale("log")
chart.set_xscale("log")
chart.grid(True, which="both", ls="--", c='gray', alpha=0.3)
chart.set_ylabel("Wahba's error (log)", fontsize=30)
chart.set_xlabel("Mean measurement noise (" + r'$\sf{\bar{\sigma}}$' + ")", fontsize=30)
chart.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.legend(prop={"size": 20})
plt.show()
