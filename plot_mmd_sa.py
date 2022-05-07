import matplotlib.pyplot as plt
import numpy as np

benign_mmd = {
    "SST2": 0.091,
    "IMDB": 0.364,
    "AMAZON": 0.106,
    "YELP": 0.212,
    "APPS": 0.139,
    "FINANCE": 0.059,
    "TWEET": 0.000,
    "RANDOM": 0.134,
    "WIKI": 0.029,
    "BENIGN": 0.045
}

sorted_benign_mmd = dict(sorted(benign_mmd.items(), key=lambda x: x[1], reverse=True))
'''
print(sorted_benign_mmd)
print(list(sorted_benign_mmd.keys()))
print(list(sorted_benign_mmd.values()))
'''

sorted_benign_mmd_key = list(sorted_benign_mmd.keys())
sorted_benign_mmd_value = list(sorted_benign_mmd.values())

fig, ax = plt.subplots(figsize=(18, 6))

mmd_ = ax.barh(sorted_benign_mmd_key, sorted_benign_mmd_value, label="Benign Query")

ax.set_xlabel("Maximum Mean Discrepancy", fontsize=16)
ax.set_title("MMD between TWEET sentiment training data and each of other SA datasets (benign query)", fontsize=18)
ax.set_yticklabels(sorted_benign_mmd_key, fontsize=16)
ax.legend()
leg = ax.legend(prop={"size":16})

ax.bar_label(mmd_, fontsize=16, padding=3)

fig.tight_layout()

plt.show()
