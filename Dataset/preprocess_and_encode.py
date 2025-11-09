"""
preprocess_and_encode.py
---------------------------------
Preprocesses 3D CT volumes and extracts deep features using the VISTA3D encoder.
Applies adaptive 3D pooling and flattens the features into (1, 4096, 768) tensors.

Usage example:
    python preprocess_and_encode.py \
        --input "train_raw" \
        --output "after_encoder" \
        --bundle "C:\\path\\to\\vista3dbundle"
"""

import os
import torch
import torch.nn.functional as F
from monai.bundle import ConfigParser
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Spacingd, CropForegroundd, ScaleIntensityRanged,
    Orientationd, CastToTyped
)
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm
import argparse


# ==============================================================
# ARGUMENT PARSER
# ==============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CTs and extract VISTA3D encoder features.")
    parser.add_argument("--input", required=True, help="Folder containing .nii.gz CT volumes.")
    parser.add_argument("--output", required=True, help="Folder to save pooled encoder outputs (.pt).")
    parser.add_argument("--bundle", required=True, help="Path to the VISTA3D bundle directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu).")
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 16),
                        help="Target 3D grid size for adaptive pooling.")
    return parser.parse_args()


# ==============================================================
# MAIN
# ==============================================================
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Input folder: {args.input}")
    print(f"[INFO] Output folder: {args.output}")
    print(f"[INFO] Bundle path: {args.bundle}")
    print(f"[INFO] Pooling grid: {args.grid}\n")

    # ----------------------------------------------------------
    # Load VISTA3D Encoder
    # ----------------------------------------------------------
    parser = ConfigParser()
    parser.read_config(os.path.join(args.bundle, "configs", "inference.json"))
    parser["bundle_root"] = args.bundle

    model = parser.get_parsed_content("network_def")
    state_dict = torch.load(os.path.join(args.bundle, "models", "model.pt"), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    encoder = model.image_encoder.encoder.to(device)

    def encoder_wrapper(x):
        feats = encoder(x)
        return feats[-1]

    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 128),
        overlap=0.3,
        sw_batch_size=1,
        mode="gaussian"
    )

    # ----------------------------------------------------------
    # Preprocessing pipeline
    # ----------------------------------------------------------
    pre = Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        CropForegroundd(keys="image", source_key="image", allow_smaller=True, margin=10),
        ScaleIntensityRanged(keys="image", a_min=-963.82, a_max=1053.67,
                             b_min=0, b_max=1, clip=True),
        Orientationd(keys="image", axcodes="RAS"),
        CastToTyped(keys="image", dtype=torch.float32),
    ])

    # ----------------------------------------------------------
    # Process files
    # ----------------------------------------------------------
    nii_files = [f for f in os.listdir(args.input) if f.endswith(".nii.gz")]
    print(f"[INFO] Found {len(nii_files)} files to process.\n")

    for name in tqdm(sorted(nii_files), desc="Processing"):
        input_path = os.path.join(args.input, name)
        output_path = os.path.join(args.output, name.replace(".nii.gz", ".pt"))

        if os.path.exists(output_path):
            continue

        try:
            # Preprocess
            sample = pre({"image": input_path})
            ct = sample["image"].unsqueeze(0).to(device, non_blocking=True)

            # Encode
            with torch.no_grad():
                features = inferer(inputs=ct, network=encoder_wrapper)  # (1, 768, D, H, W)

                # Adaptive Pool + Flatten
                pooled = F.adaptive_avg_pool3d(features, args.grid)  # (1, 768, 16, 16, 16)
                out_tokens = pooled.flatten(2).transpose(1, 2).contiguous()  # (1, 4096, 768)
                torch.save(out_tokens.half().cpu(), output_path)

            print(f"[INFO] Saved {name} → {tuple(out_tokens.shape)}")

            del features, pooled, out_tokens, ct
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] Failed {name}: {e}")

    print("\nAll volumes processed successfully!")


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    main()
