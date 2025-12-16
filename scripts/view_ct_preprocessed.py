"""
view_ct_preprocessed.py
---------------------------------
Loads and preprocesses a raw CT scan from Dataset/raw_ct_scans/
using the same pipeline as VISTA3D, and provides a scrollable
slice viewer for easy inspection.

Usage:
    python scripts/view_ct_preprocessed.py --file "train_8429_a_1.nii.gz"
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Spacingd, CropForegroundd, ScaleIntensityRanged,
    Orientationd, CastToTyped
)


# ==============================================================
# ARGUMENTS
# ==============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and view a CT scan interactively.")
    parser.add_argument("--file", required=True, help="Name of the CT file inside Dataset/raw_ct_scans/")
    return parser.parse_args()


# ==============================================================
# MAIN
# ==============================================================
def main():
    args = parse_args()

    # Resolve project base and CT path (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ct_path = os.path.join(project_root, "Dataset", "raw_ct_scans", args.file)

    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    print(f"[INFO] Loading CT: {ct_path}")

    # ----------------------------------------------------------
    # Preprocessing 
    # ----------------------------------------------------------
    pre = Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        CropForegroundd(keys="image", source_key="image", allow_smaller=True, margin=10),
        ScaleIntensityRanged(keys="image", a_min=-964, a_max=1054, b_min=0, b_max=1, clip=True),
        Orientationd(keys="image", axcodes="RAS"),
        CastToTyped(keys="image", dtype=torch.float32),
    ])

    # Run pipeline
    out = pre({"image": ct_path})
    ct = out["image"].squeeze().cpu().numpy()  # shape: (D, H, W)
    print(f"[INFO] Preprocessed shape: {ct.shape}")

    # Scrollable Viewer

    class ScrollViewer:
        def __init__(self, ct):
            self.ct = ct
            self.slices = ct.shape[2]
            self.ind = self.slices // 2

            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.im = self.ax.imshow(np.rot90(self.ct[:, :, self.ind]), cmap='gray')
            self.ax.set_title(f'CT Slice {self.ind + 1}/{self.slices}')
            self.ax.axis('off')

            # Events
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)

            plt.tight_layout()
            plt.show()

        def update(self):
            self.im.set_data(np.rot90(self.ct[:, :, self.ind]))
            self.ax.set_title(f'CT Slice {self.ind + 1}/{self.slices}')
            self.fig.canvas.draw_idle()

        def on_scroll(self, event):
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            elif event.button == 'down':
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def on_key(self, event):
            if event.key == 'right':
                self.ind = (self.ind + 1) % self.slices
            elif event.key == 'left':
                self.ind = (self.ind - 1) % self.slices
            self.update()

    ScrollViewer(ct)

if __name__ == "__main__":
    main()
