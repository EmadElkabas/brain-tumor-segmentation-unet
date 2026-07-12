
# Brain MRI Tumor Segmentation with U-Net

Segments tumor regions in brain MRI scans using a U-Net built from scratch in PyTorch, benchmarked against a pretrained-encoder U-Net (ResNet34/ImageNet).

## Dataset

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

— 110 patients, ~3,900 MRI slices with paired tumor masks, via Kaggle.  
Split by patient (not by slice) to prevent data leakage between train/val/test.

## Approach

- **U-Net (from scratch)**: encoder-decoder with skip connections, implemented and trained from random initialization.

- **U-Net (pretrained)**: same architecture shape, ResNet34 encoder pretrained on ImageNet (`segmentation_models_pytorch`).

- **Loss**: BCE + Dice combined (handles class imbalance — tumors are a small fraction of each image).

- **Augmentation**: horizontal flip, rotation (±15°), brightness/contrast — chosen for anatomical plausibility (no vertical flip, since brain scans have a fixed up/down orientation).

- **Postprocessing**: removes predicted blobs under 50px — cuts false positive "hallucinated" detections on tumor-free slices.

- **Training**: 50 epochs, ReduceLROnPlateau scheduler, Adam optimizer.

## Results (test set)

| Model | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| **U-Net (from scratch)** | **0.7738** | **0.6638** | 0.8223 | **0.7799** |
| U-Net (pretrained ResNet34) | 0.7472 | 0.6396 | **0.8499** | 0.7294 |

## Key findings

- **From-scratch outperformed the pretrained encoder** on Dice/IoU/Recall after 50 epochs, despite pretrained converging faster early on. ImageNet features (natural photos) don't perfectly transfer to grayscale MRI texture — given enough epochs, a domain-specific model can catch up and exceed a generic pretrained one.

- Augmentation reduced overfitting (train/val loss gap shrank) and meaningfully improved Dice (+8 points) over the unaugmented baseline.

- Raw predictions showed false-positive "hallucinated" tumors on tumor-free slices; a simple blob-size postprocessing filter reduced this without costing meaningful recall.

- Pixel accuracy is a misleading metric (>99% even for a naive model) due to tumors covering a small fraction of each image — Dice/IoU are the metrics that actually matter for this task.

## Files

- `notebooks/brain-tumor-segmentation-unet.ipynb` — complete training and evaluation notebook
- `results/` — prediction visualization examples
- `src/` — source code
- `tests/` — testing files

## Setup

```bash
pip install -r requirements.txt

## Credits
 Dataset: [LGG MRI Segmentation]
 (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Buda et al.). Pretrained encoder via [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch). 
 
