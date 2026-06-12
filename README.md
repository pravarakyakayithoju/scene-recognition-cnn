# 🏔️ Scene Recognition with a Convolutional Neural Network

A CNN that classifies photographs of natural and urban scenes into **6 categories**: buildings, forest, glacier, mountain, sea, and street.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **~88%** (fill with your number) |
| **Classes** | 6 scene types |
| **Test Images** | ~3,000 |

Most confused pair: **glacier ↔ mountain** — a reasonable confusion even for humans (snow-covered peaks vs ice fields).

---

## 🧠 How It Works

### CNN Architecture

```
Input (100×100×3 image)
    ↓
Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout(25%)
    detects: edges, color gradients, basic textures
    ↓
Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout(25%)
    detects: shapes, structures (tree canopy, wave patterns)
    ↓
Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(25%)
    detects: scene-level features (mountain ridges, urban perspective)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(50%) → Dense(6, softmax)
    output: probability for each scene type
```

### Key Design Decisions

- **Data augmentation** (rotation, shift, zoom, horizontal flip) — scenes appear at varying angles and distances. Horizontal flip is safe here (a forest flipped is still a forest — unlike traffic signs where flipping changes meaning).
- **3 conv blocks with increasing filters** (32 → 64 → 128) — each block captures more complex visual abstractions.
- **BatchNormalization** after each conv — stabilizes training.
- **Dropout (25% in conv, 50% before output)** — prevents overfitting, especially important since some classes have fewer images.

---

## 📊 What the Notebook Covers

1. Load and resize 25,000+ images from class-labeled folders
2. Visualize samples and check class distribution
3. Data augmentation with domain-specific choices
4. Build and explain a 3-block CNN architecture
5. Train with validation monitoring (15 epochs, ~10-15 min on CPU)
6. Training curves — accuracy and loss over epochs
7. Confusion matrix and per-class precision/recall/F1
8. Visual predictions with correct/wrong labels
9. Per-class accuracy analysis — hardest and easiest scenes

---

## 🛠️ Tech Stack

TensorFlow/Keras, OpenCV, scikit-learn, NumPy, matplotlib, seaborn

---

## 📦 Dataset

**Intel Image Classification** (Kaggle)
Download: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

- 6 classes: buildings, forest, glacier, mountain, sea, street
- ~14,000 training + ~3,000 test images
- Real-world 150×150 photographs (resized to 100×100 for training)

---

## ⚙️ How to Run

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

1. Download the dataset from Kaggle, extract it
2. Place the extracted folder next to the notebook (adjust `DATA_DIR` in cell 2 if needed)
3. Run top to bottom (~10-15 minutes on CPU)

---

## 💡 Possible Extensions

- Transfer learning with pretrained ResNet/EfficientNet (~93%+ accuracy)
- Grad-CAM visualization to see which image regions the CNN focuses on
- Deploy as a Streamlit web app for real-time scene classification

---

*ML portfolio — deep learning & computer vision.*
