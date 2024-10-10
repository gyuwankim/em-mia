import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def draw_score_distributions(bins, hists, fig_path=None, score_name="ReCaLL"):
    _, ax = plt.subplots(figsize=(6, 4))
    for hist, label, color in zip(hists, ("non-member", "member"), ("orange", "skyblue")):
        ax.stairs(hist, bins, label=label, fill=True, alpha=.5, color=color)
    ax.set_xlabel(f"{score_name} Score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    # plt.show()
    if fig_path:
        plt.savefig(fig_path, dpi=300)
    plt.close()    


def plot_roc_curves(roc_curves, fig_path=None):
    _, ax = plt.subplots(figsize=(4, 3))
    ax.plot([], [], "w", linewidth=0, label="Method (AUC-ROC)")
    for method, (fpr, tpr) in roc_curves.items():
        ax.plot(fpr, tpr, linewidth=0.2, label=f"{method} ({auc(fpr, tpr) * 100.:.1f})")
    ax.plot([0, 1], [0, 1], "k-", linewidth=0.2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=6, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, num=6, endpoint=True))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=2)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    # plt.show()
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
