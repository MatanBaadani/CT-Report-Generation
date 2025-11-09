import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG — relative paths
# =====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints/train_log.txt")
SAVE_PATH = os.path.join(PROJECT_ROOT, "loss_plot.pdf")  # <-- high-quality PDF for LyX

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(LOG_PATH)

# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(7, 4.5))  # good proportion for LyX

plt.plot(df["epoch"], df["train_loss"], label="Training Loss", linewidth=2.2, color="#1f77b4")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2.2, color="#ff7f0e")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=13, weight="bold")

plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# =====================================================
# SAVE — vector PDF (best for LyX)
# =====================================================
plt.savefig(SAVE_PATH, format="pdf", dpi=600, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved high-quality PDF to {SAVE_PATH}")
