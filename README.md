# Cancer detection

A Streamlit web app that classifies medical images from seven cancer areas (brain, breast, cervical, kidney, lung/colon, lymphoma and oral) and shows a Grad-CAM heatmap of the regions the model used. Each area has its own Keras model, built by transfer learning on MobileNetV3Large and trained on the Kaggle Multi-Cancer dataset. It is a personal project.

Live demo: [cancerdetection-lewis.streamlit.app](https://cancerdetection-lewis.streamlit.app/)

![Grad-CAM heatmaps over brain MRI scans](Grad%20Cam%20Brain%20Scan%20Fig%2012.png)

## How it works

1. Training (`model_creation/model_creation.ipynb`) loads each cancer area from the [Multi-Cancer dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer) with `kagglehub`, resizes images to 224x224 and splits them 80/20 into training and validation sets with a fixed seed.
2. Each model is MobileNetV3Large with ImageNet weights and a frozen base, followed by global average pooling, dense layers of 1024, 512 and 128 units (ReLU) and a softmax output. It is trained with Adam (learning rate 0.0001) and categorical cross-entropy for 10 epochs (15 for oral).
3. In the app (`home.py`) the user uploads an image and picks the cancer area. The matching model returns a probability for every class in that area.
4. Grad-CAM takes gradients of the predicted class with respect to the `conv_1_bn` layer, averages them per channel, weights the feature maps and overlays the result on the image with a jet colour map.

The classes per area are: brain (glioma, meningioma, tumour), breast (benign, malignant), cervical (five cell types), kidney (normal, tumour), lung/colon (colon adenocarcinoma, colon benign, lung adenocarcinoma, lung benign, lung squamous cell carcinoma), lymphoma (CLL, follicular, mantle cell) and oral (normal, squamous cell carcinoma).

## Results

Accuracy on the 20% validation split, from `model.evaluate` in the training notebook:

| Cancer area | Validation images | Accuracy |
|---|---|---|
| Brain | 3,000 | 99.47% |
| Breast | 2,000 | 99.70% |
| Cervical | 5,000 | 99.96% |
| Kidney | 2,000 | 100.00% |
| Lung/colon | 5,000 | 99.70% |
| Lymphoma | 3,000 | 98.10% |
| Oral | 2,000 | 93.85% |

There is no separate test set, so these figures are likely to be optimistic.

## Repository layout

| Path | What it is |
|---|---|
| `home.py` | Streamlit app: detection page, demo page with sample images, and background pages on the model and dataset |
| `models/` | The seven trained `.keras` models (about 31 MB each) loaded by the app |
| `model_creation/model_creation.ipynb` | Data loading, training, evaluation, confusion matrix and sample predictions |
| `grad_cam.py` | Batch script that writes Grad-CAM overlays for every image in `images/` to `grad_cam_outputs/` |
| `images/`, `grad_cam_outputs/` | Brain and breast sample images used by the demo page, and their Grad-CAM overlays |
| `mammography_vs_nhs.ipynb` | Follow-up experiment: a breast cancer model on RSNA screening mammograms with a patient-level split, compared with NHS first-reader sensitivity at matched specificity |
| `pdf.py` | Prototype ReportLab script that lays out a PDF analysis report |
| `preprocess.py` | Earlier standalone version of the prediction function |
| `MobileNetV3 Architecture.png` | Diagram of the MobileNetV3 backbone |
| `requirements.txt`, `packages.txt`, `.devcontainer/` | Python packages, system packages for Streamlit Cloud, and a Codespaces config (Python 3.11) |

## Running it

```bash
git clone https://github.com/lewis-shearer/Cancer_Detection.git
cd Cancer_Detection
pip install -r requirements.txt
streamlit run home.py
```

To regenerate the Grad-CAM samples, run `python grad_cam.py`. Retraining needs the Multi-Cancer dataset, which the notebook downloads with `kagglehub`; the dataset path in the notebook is hard-coded to the Codespaces cache directory. The mammography notebook needs the RSNA Screening Mammography competition rules accepted on Kaggle and a GPU.

## Limitations

- This is not a medical tool and should not be used for diagnosis.
- The train/validation split is random per image, and the validation set was the only held-out data.
- Each model always answers with one of its own classes, so the result is meaningless if the wrong cancer area is chosen or the image is not from that dataset.
- `mammography_vs_nhs.ipynb` has not been run yet and has no results.
- `pdf.py` uses hard-coded placeholder data, is not connected to the app and needs Arial TTF files on the system.
- The demo page only has brain and breast sample images.
