# CT Report Generation

This project implements a BLIP-2–style architecture for **automatic radiology report generation** directly from **3D chest CT scans**.  
It demonstrates how powerful pretrained models can be reused in the medical domain to generate structured, clinically consistent reports with limited data and computation.

---

#Project Overview

The model combines:
- **VISTA3D Encoder** – a frozen 3D SegResNet-based vision backbone pretrained for medical segmentation  
- **Q-Former** – a lightweight transformer bridging visual and textual modalities through cross-attention  
- **T5-Small Decoder** – a language model that generates the final radiology report  

Only the Q-Former and decoder are trained, while the encoder remains frozen and used only once to pre-extract features.

---

Dataset

The project uses the **CT-RATE dataset** of chest CT volumes paired with English radiology reports.  
Due to compute and time limits, training was conducted on approximately **6,000 scans** out of the ~50,000 available in the dataset.  
All CTs were preprocessed through the VISTA3D pipeline (resampling, orientation to RAS, HU clipping, and adaptive pooling to 16×16×16 features).

---

#Training Details

- **Encoder:** Frozen (VISTA3D)
- **Q-Former:** 4 layers, 8 heads, 128 query tokens
- **Decoder:** T5-Small (trained in phases)
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Loss:** Cross-entropy on report tokens  
- **Precision:** Mixed FP16 training  
- **Epochs:** 40  
- **Batch Size:** 1 (with gradient accumulation = 4)  

Unfreezing schedule:
1. Epoch 1–5: Q-Former only  
2. Epoch 5–15: + top-2 decoder layers  
3. Epoch 15–50: full decoder fine-tuning  

---

 #Evaluation

Evaluation metrics:
- **BLEU-4** – text overlap precision  
- **ROUGE-L** – structural similarity  
- **CXR-BERT F1** – clinically aware evaluation using a pretrained classifier  

Validation is run every 5 epochs; the checkpoint with the lowest validation loss is selected for testing.

---

#Results Summary

Even though the model was trained on a small subset (≈6k scans), it successfully captured major clinical patterns such as:
- Airway and mediastinal structure descriptions  
- Coronary calcifications  
- COVID-19–related ground-glass opacities  

Some fine spatial findings (e.g., nodule count or region) remain less precise, showing room for improvement.



Resources and Setup

Before running the project, make sure the following components are available:

1) Dataset
Download the CT-RATE dataset from Hugging Face:
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

This dataset contains paired 3D chest CT volumes and English radiology reports used for training and evaluation.

2) VISTA3D Encoder (Pretrained 3D Backbone)
Download the pretrained VISTA3D bundle provided by MONAI:
https://github.com/Project-MONAI/VISTA/tree/main/vista3d

The model is used in frozen mode for 3D feature extraction during preprocessing.

3) T5 Text Decoder
Use the pretrained T5-Small model from Hugging Face:
https://huggingface.co/google-t5/t5-small

This serves as the language decoder in the BLIP-2–style architecture and can be fine-tuned using the provided training script.

4) CXR-BERT Model for Clinical F1 Evaluation
To compute the Clinical F1 metric, use or fine-tune the CXR-BERT model trained for chest report classification:
https://huggingface.co/StanfordAIMI/CXR-BERT-general

This model predicts common radiological findings from text and is applied to both ground-truth and generated reports to assess clinical consistency.

---

 #Citation

If you use or adapt this project, please cite the following works:

-Hamamci et al., CT2Rep: 3D Chest CT to Radiology Report Generation, 2024

-Li et al., BLIP-2: Bootstrapping Language-Image Pre-training, 2023

-Wang et al., VISTA3D: Versatile Imaging Segmentation Through Vision Transformers, 2024

-Raffel et al., T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, 2020


Author

Matan Baadani
Implementation, experimentation, and documentation
Tel Aviv University – Deep Learning Course, 2025


