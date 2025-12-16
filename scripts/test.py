import os, torch, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support

# ----------------------------------------------------------------------
# LOCAL IMPORTS
# ----------------------------------------------------------------------
from models.qformer_t5_bridge import QFormerT5, BridgeConfig


# ==============================================================
# CONFIG 
# ==============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.path.join(DATASET_DIR, "test_pooled_data")
CSV_PATH = os.path.join(DATASET_DIR, "train_reports.csv")
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "best.pt")
F1_MODEL_DIR = os.path.join(DATASET_DIR, "f1_score")

SAVE_RESULTS = os.path.join(PROJECT_ROOT, "test_results_classifier_f1.csv")

T5_NAME = "t5-small"
MAX_TEXT_LEN = 384
BATCH_SIZE = 1


# ==============================================================
# DATASET
# ==============================================================
class CTFeatureTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, feature_dir, tokenizer, max_length=256):
        df = pd.read_csv(csv_path)
        valid = []
        for _, row in df.iterrows():
            name = row["VolumeName"].replace(".nii.gz", "")
            feat_path = os.path.join(feature_dir, f"{name}.pt")
            if os.path.isfile(feat_path):
                valid.append({"feat": feat_path, "text": str(row["Findings_EN"])})
        self.samples = valid
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[INFO] Loaded {len(self.samples)} test examples from {feature_dir}.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        feat = torch.load(ex["feat"]).to(torch.float32)
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        tok = self.tokenizer(
            ex["text"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tok["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return feat, labels, ex["text"]


# ==============================================================
# BLEU + ROUGE
# ==============================================================
def compute_bleu_rouge(pred, ref):
    smoothie = SmoothingFunction().method4
    bleu4 = sentence_bleu([ref.split()], pred.split(),
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothie)
    rougeL = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)["rougeL"].fmeasure
    return bleu4, rougeL


# ==============================================================
# CLINICAL F1
# ==============================================================
device_f1 = DEVICE
tokenizer_f1 = AutoTokenizer.from_pretrained(F1_MODEL_DIR)
model_f1 = AutoModelForSequenceClassification.from_pretrained(F1_MODEL_DIR).to(device_f1)
with open(os.path.join(F1_MODEL_DIR, "labels.json"), "r") as f:
    LABELS = json.load(f)["labels"]

def predict_logits(text):
    enc = tokenizer_f1(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    ).to(device_f1)
    with torch.no_grad():
        logits = model_f1(**enc).logits.cpu().numpy()
    return logits

def logits_to_labels(logits, threshold=0.5):
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= threshold).astype(int)

def compute_clinical_f1(gt_text, gen_text):
    logit_gt = predict_logits(gt_text)
    logit_gen = predict_logits(gen_text)
    y_true = logits_to_labels(logit_gt)[0]
    y_pred = logits_to_labels(logit_gen)[0]
    micro = precision_recall_fscore_support([y_true], [y_pred], average="micro", zero_division=0)
    return micro[2]  # return F1 only


# ==============================================================
# MAIN
# ==============================================================
def main():
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    ds = CTFeatureTextDataset(CSV_PATH, DATA_DIR, tokenizer, max_length=MAX_TEXT_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


    bridge_cfg = BridgeConfig(t5_name=T5_NAME, freeze_t5_encoder=False, freeze_t5_decoder=False)
    model = QFormerT5(
        vision_dim=768, q_hidden_dim=512, num_query_tokens=128,
        q_depth=4, q_heads=8, kv_posenc="sinusoidal", bridge_cfg=bridge_cfg
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")

    total_loss = bleu_sum = rouge_sum = f1_sum = 0.0
    count = 0
    results = []

    with torch.no_grad():
        for feats, labels, refs in tqdm(dl, desc="Evaluating"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            attn_mask = torch.ones(feats.shape[:2], dtype=torch.long, device=DEVICE)

            out = model(vision_feats=feats, vision_mask=attn_mask, labels=labels)
            loss = out.loss.item()
            total_loss += loss

            gen_ids = model.generate(
                vision_feats=feats,
                vision_mask=attn_mask,
                max_new_tokens=256,
                num_beams=3,
                no_repeat_ngram_size=3,
                length_penalty=0.6,
                early_stopping=True,
            )
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            ref = refs[0]

            bleu4, rougeL = compute_bleu_rouge(pred, ref)
            clinical_f1 = compute_clinical_f1(ref, pred)

            results.append({
                "Prediction": pred,
                "Reference": ref,
                "Loss": loss,
                "BLEU-4": bleu4,
                "ROUGE-L": rougeL,
                "ClinicalF1": clinical_f1,
            })

            bleu_sum += bleu4
            rouge_sum += rougeL
            f1_sum += clinical_f1
            count += 1

    avg_loss = total_loss / count
    avg_bleu = bleu_sum / count
    avg_rouge = rouge_sum / count
    avg_f1 = f1_sum / count

    print("\n==== Final Test Results ====")
    print(f"Loss       : {avg_loss:.4f}")
    print(f"BLEU-4     : {avg_bleu:.4f}")
    print(f"ROUGE-L    : {avg_rouge:.4f}")
    print(f"Clinical F1: {avg_f1:.4f}")

    pd.DataFrame(results).to_csv(SAVE_RESULTS, index=False)
    print(f"[INFO] Saved detailed results to {SAVE_RESULTS}")


if __name__ == "__main__":
    main()
