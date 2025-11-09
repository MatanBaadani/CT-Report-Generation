from huggingface_hub import hf_hub_download
import pandas as pd
import os
import shutil

repo_id = "ibrahimhamamci/CT-RATE"
directory_name = "dataset/train_fixed/"

# where you want to save the flat structure
output_dir = os.path.join(os.path.dirname(__file__), "raw_ct_scans")
os.makedirs(output_dir, exist_ok=True)

# path to your CSV file with VolumeName
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "train_reports.csv"))

# ✨ Add these two variables:
start_index =0  # where to start (e.g. 0 for first batch, 1000 for second, etc.)
max_files = 6000   # how many files to download in this run

# slice the dataframe for the batch you want
subset = data["VolumeName"].iloc[start_index:start_index + max_files]

for i, name in enumerate(subset, start=start_index):
    folder1 = name.split("_")[0]
    folder2 = name.split("_")[1]
    folder = folder1 + "_" + folder2
    folder3 = name.split("_")[2]
    subfolder = folder + "_" + folder3
    subfolder = directory_name + folder + "/" + subfolder

    # Download to a temp location (original structure)
    temp_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        subfolder=subfolder,
        filename=name,
        local_dir="tmp_download"
    )

    # Construct flat output path (filename only)
    flat_path = os.path.join(output_dir, name)

    # Move file to flat directory
    shutil.move(temp_path, flat_path)

    print(f"Downloaded and moved: {i} - {name}")
