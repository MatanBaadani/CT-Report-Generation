import os, random, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

from transformers import T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
from models.qformer_t5_bridge import QFormerT5, BridgeConfig

# ============================================================
# CONFIG 
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = os.path.join(DATASET_DIR, "train_pooled_data")
VAL_DIR   = os.path.join(DATASET_DIR, "val_pooled_data")

TRAIN_CSV = os.path.join(DATASET_DIR, "train_reports.csv")
F1_MODEL_DIR = os.path.join(DATASET_DIR, "f1_score")

T5_NAME = "t5-small"
EPOCHS = 50
BATCH_SIZE = 1
ACCUM_STEPS = 4
LR_QFORMER = 1e-4
LR_T5_DEC = 5e-5
MAX_TEXT_LEN = 384

SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_PATH = os.path.join(SAVE_DIR, "train_log.txt")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================
class CTFeatureTextDataset(Dataset):
    def __init__(self, feature_dir, csv_path, tokenizer, max_length=256):
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)
        if "VolumeName" not in df.columns or "Findings_EN" not in df.columns:
            raise ValueError("CSV must contain VolumeName + Findings_EN columns")

        self.report_map = {
            str(row["VolumeName"]).replace(".nii.gz","").replace(".pt",""): str(row["Findings_EN"])
            for _, row in df.iterrows()
        }

        self.pt_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".pt")])
        if len(self.pt_files) == 0:
            raise FileNotFoundError(f"No .pt feature files found in {feature_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        fname = self.pt_files[idx]
        stem = fname.replace(".pt","")

        feat_path = os.path.join(self.feature_dir, fname)
        feat = torch.load(feat_path)
        if feat.dim()==3 and feat.shape[0]==1:
            feat = feat.squeeze(0)
        feat = feat.to(torch.float32)

        if stem not in self.report_map:
            raise KeyError(f"VolumeName {stem} not found in CSV {self.feature_dir}")

        text = self.report_map[stem]
        tok = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tok["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return feat, labels


# ============================================================
# METRICS
# ============================================================
def compute_bleu4(pred, ref):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref.split()], pred.split(),
                         weights=(0.25,0.25,0.25,0.25),
                         smoothing_function=smoothie)

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def compute_rougeL(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure

class ClinicalF1Scorer:
    def __init__(self, model_dir, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        with open(os.path.join(model_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)["labels"]

    @torch.no_grad()
    def logits(self, texts):
        enc = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits.cpu().numpy()
        return logits

    @staticmethod
    def binarize(logits, thr=0.5):
        probs = 1/(1+np.exp(-logits))
        return (probs >= thr).astype(int)

    def micro_f1(self, true_txts, pred_txts, thr=0.5):
        y_true = self.binarize(self.logits(true_txts), thr)
        y_pred = self.binarize(self.logits(pred_txts), thr)
        p,r,f,_ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        return p, r, f


# ============================================================
# TRAINING
# ============================================================
def main():

    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)

    # datasets
    train_ds = CTFeatureTextDataset(TRAIN_DIR, TRAIN_CSV, tokenizer, MAX_TEXT_LEN)
    val_ds   = CTFeatureTextDataset(VAL_DIR, TRAIN_CSV, tokenizer, MAX_TEXT_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False)

    # model
    bridge_cfg = BridgeConfig(t5_name=T5_NAME, freeze_t5_encoder=True, freeze_t5_decoder=False)
    model = QFormerT5(
        vision_dim=768,
        q_hidden_dim=512,
        num_query_tokens=128,
        q_depth=4,
        q_heads=8,
        kv_posenc="sinusoidal",
        bridge_cfg=bridge_cfg
    ).to(DEVICE)

    # Phase 1: Q-Former only
    for p in model.t5.parameters():
        p.requires_grad = False

    opt = AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_QFORMER}],
                weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    clin_scorer = ClinicalF1Scorer(F1_MODEL_DIR, DEVICE)

    with open(LOG_PATH, "w") as f:
        f.write("epoch,phase,train_loss,val_loss,bleu4,rougeL,clinicalF1\n")

    def phase(e):
        if e < 5: return "QFormer-only"
        if e < 15: return "TopDecoder"
        return "FullDecoder"

    print(f"[INFO] Train={len(train_ds)} | Val={len(val_ds)}")

    for epoch in range(EPOCHS):
        # Phase unfreezing
        if epoch == 5:
            print("[INFO] Phase 2: unfreeze top-2 decoder blocks")
            for p in model.t5.decoder.block[-2:].parameters():
                p.requires_grad = True
            opt = AdamW([
                {"params": model.qformer.parameters(), "lr": LR_QFORMER},
                {"params": model.t5.decoder.block[-2:].parameters(), "lr": LR_T5_DEC}
            ], weight_decay=0.01)
            sched = CosineAnnealingLR(opt, T_max=EPOCHS-epoch)

        if epoch == 15:
            print("[INFO] Phase 3: unfreeze full decoder")
            for p in model.t5.decoder.parameters():
                p.requires_grad = True
            opt = AdamW([
                {"params": model.qformer.parameters(), "lr": LR_QFORMER},
                {"params": [p for n,p in model.t5.named_parameters() if "decoder" in n and p.requires_grad],
                 "lr": LR_T5_DEC}
            ], weight_decay=0.01)
            sched = CosineAnnealingLR(opt, T_max=EPOCHS-epoch)

        # TRAIN

        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}")
        for i, (feats, labels) in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            vm = torch.ones(feats.shape[0], feats.shape[1], dtype=torch.long, device=DEVICE)

            with torch.amp.autocast("cuda"):
                out = model(vision_feats=feats, vision_mask=vm, labels=labels)
                loss = out.loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i+1) % ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss) * ACCUM_STEPS
            pbar.set_postfix(loss=total_loss / (i+1))

        sched.step()
        train_loss = total_loss / len(train_dl)
        print(f"[TRAIN] Epoch {epoch} | loss={train_loss:.4f}")

        # VALIDATION LOSS (every epoch)

        model.eval()
        vloss_sum = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for feats, labels in val_dl:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                vm = torch.ones(feats.shape[0], feats.shape[1], dtype=torch.long, device=DEVICE)
                out = model(vision_feats=feats, vision_mask=vm, labels=labels)
                vloss_sum += out.loss.item()
        val_loss = vloss_sum / len(val_dl)
        print(f"[VAL LOSS] Epoch {epoch} | {val_loss:.4f}")

        bleu4 = rougeL = clinF1 = 0.0

        # EVALUATION (every 5 epochs)
        
        if (epoch+1) % 5 == 0 or epoch == EPOCHS-1:
            refs_all, preds_all = [], []
            bsum, rsum = 0.0, 0.0
            for feats, labels in tqdm(val_dl, desc="Val Metrics"):
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                vm = torch.ones(feats.shape[0], feats.shape[1], dtype=torch.long, device=DEVICE)
                gen = model.generate(
                    vision_feats=feats, vision_mask=vm,
                    max_new_tokens=256, num_beams=3,
                    no_repeat_ngram_size=3, length_penalty=0.6,
                    early_stopping=True
                )
                pred = tokenizer.decode(gen[0], skip_special_tokens=True)
                ref  = tokenizer.decode(labels[0][labels[0]!=-100], skip_special_tokens=True)
                bsum += compute_bleu4(pred, ref)
                rsum += compute_rougeL(pred, ref)
                refs_all.append(ref)
                preds_all.append(pred)

            bleu4 = bsum / len(val_dl)
            rougeL = rsum / len(val_dl)
            _, _, clinF1 = clin_scorer.micro_f1(refs_all, preds_all)
            print(f"[VAL METRICS] Epoch {epoch} | BLEU-4={bleu4:.4f}  ROUGE-L={rougeL:.4f}  F1={clinF1:.4f}")

            # Example print
            idx = random.randint(0, len(val_ds)-1)
            sfeat, slabel = val_ds[idx]
            sfeat, slabel = sfeat.unsqueeze(0).to(DEVICE), slabel.unsqueeze(0).to(DEVICE)
            vm = torch.ones(sfeat.shape[0], sfeat.shape[1], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                gen = model.generate(
                    vision_feats=sfeat, vision_mask=vm,
                    max_new_tokens=256, num_beams=3,
                    no_repeat_ngram_size=3, length_penalty=0.6,
                    early_stopping=True
                )
            pred_s = tokenizer.decode(gen[0], skip_special_tokens=True)
            ref_s  = tokenizer.decode(slabel[0][slabel[0]!=-100], skip_special_tokens=True)
            print("\n[Sample from Validation Set]")
            print("GT :", ref_s)
            print("GEN:", pred_s, "\n")

            ckpt = os.path.join(SAVE_DIR, f"epoch_{epoch:02d}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"[CKPT] saved {ckpt}")

        # LOGGING
        with open(LOG_PATH, "a") as f:
            f.write(f"{epoch},{phase(epoch)},{train_loss:.4f},{val_loss:.4f},"
                    f"{bleu4:.4f},{rougeL:.4f},{clinF1:.4f}\n")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
