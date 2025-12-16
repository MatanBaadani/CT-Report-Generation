

import os
import random
import shutil


# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "after_encoder_and_pooled")  # Folder created by preprocess_and_encode.py
TRAIN_DIR = os.path.join(BASE_DIR, "train_pooled_data")
VAL_DIR = os.path.join(BASE_DIR, "val_pooled_data")
TEST_DIR = os.path.join(BASE_DIR, "test_pooled_data")

# hard-coded split percentages
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# create output folders
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print(f"[INFO] Input folder: {INPUT_DIR}")
print(f"[INFO] Output folders: train={TRAIN_DIR}, val={VAL_DIR}, test={TEST_DIR}\n")


# ==============================================================
# HELPER — Extract patient ID
# ==============================================================
def get_patient_id(fname: str) -> str:
    # Example: train_8429_a_1.pt  →  "8429"
    parts = fname.split("_")
    if len(parts) > 1:
        return parts[1]
    else:
        return "unknown"


# ==============================================================
# LOAD ALL FILES
# ==============================================================
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")]
print(f"[INFO] Found {len(all_files)} encoded files.")

# Group files by patient ID
patients = {}
for f in all_files:
    pid = get_patient_id(f)
    patients.setdefault(pid, []).append(f)

print(f"[INFO] Found {len(patients)} unique patients.\n")

# shuffle patients for randomness
patient_ids = list(patients.keys())
random.shuffle(patient_ids)

#  patient counts for each split
num_patients = len(patient_ids)
num_train = int(num_patients * TRAIN_RATIO)
num_val = int(num_patients * VAL_RATIO)
num_test = num_patients - num_train - num_val

train_patients = set(patient_ids[:num_train])
val_patients = set(patient_ids[num_train:num_train + num_val])
test_patients = set(patient_ids[num_train + num_val:])

print(f"Patient split Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}\n")


# ==============================================================
# ASSIGN FILES BASED ON PATIENT SPLIT
# ==============================================================
train_files, val_files, test_files = [], [], []

for pid, files in patients.items():
    if pid in train_patients:
        train_files.extend(files)
    elif pid in val_patients:
        val_files.extend(files)
    elif pid in test_patients:
        test_files.extend(files)

print(f"[INFO] File counts Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}\n")


# ==============================================================
# COPY FILES
# ==============================================================
def copy_files(files, src_dir, dst_dir, split_name):
    print(f"[INFO] Copying {len(files)} files to {split_name}/ ...")
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    print(f"[INFO] Done copying {split_name}.\n")


copy_files(train_files, INPUT_DIR, TRAIN_DIR, "train_pooled_data")
copy_files(val_files, INPUT_DIR, VAL_DIR, "val_pooled_data")
copy_files(test_files, INPUT_DIR, TEST_DIR, "test_pooled_data")

print("Split complete! No patient appears in multiple sets.")
