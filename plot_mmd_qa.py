import matplotlib.pyplot as plt
import numpy as np

in_domain = ["NewsQA", "SearchQA", "TriviaQA", "HotpotQA", "NaturalQuestions"]
# in_domain_mmd = [0.242736, 0.091251, 0.083678, 0.054122, 0.031344]
in_domain_mmd = [0.243, 0.091, 0.084, 0.054, 0.031]

out_domain = ["DuoRC", "DROP", "BioASQ", "RACE", "TextbookQA", "RelationExtraction"]
# out_domain_mmd = [0.496178, 0.369885, 0.266987, 0.127625, 0.123036, 0.060088]
out_domain_mmd = [0.496, 0.370, 0.267, 0.128, 0.123, 0.060]

benign_mmd {
	"SearchQA": 0.091,
	"NewsQA": 0.243,
	"NaturalQuestions": 0.031,
	"TriviaQA": 0.084,
	"SQuAD": 0.000,
	"HotpotQA": 0.054,
	"RACE": 0.128,
	"DROP": 0.370,
	"BioASQ": 0.267,
	"TextbookQA": 0.123,
	"DuoRC": 0.496,
	"RelationExtraction": 0.060,
	"RANDOM": 0.403,
	"WIKI": 0.018,
	"BENIGN": 0.031
}


attack = ["RANDOM", "WIKI"]
attack_mmd = [0.403, 0.018]

benign = ["BENIGN"]
benign_ = [0.031] #0.031356

fig, ax = plt.subplots(figsize=(18, 6))

od_ = ax.barh(out_domain, out_domain_mmd, label="Out-Domain")
id_ = ax.barh(in_domain, in_domain_mmd, label="In-Domain")
benign_ = ax.barh(["BENIGN"], [0.031], label="BENIGN")

ax.set_xlabel("Maximum Mean Discrepancy", fontsize=16)
ax.set_title("MMD between SQuAD and each In-Domain/Out-Domain dataset", fontsize=18)
ax.set_yticklabels(in_domain+out_domain, fontsize=16)
ax.legend()
leg = ax.legend(prop={"size":16})

ax.bar_label(od_, fontsize=16, padding=3)
ax.bar_label(id_, fontsize=16, padding=3)
ax.bar_label(benign_, fontsize=16, padding=3)

fig.tight_layout()

plt.show()