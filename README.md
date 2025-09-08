# 🧠 **Multi-Class Retinal Disease Classification Using OCT Imaging**

### A Comprehensive CNN & Transformer Benchmark with a Novel Hybrid 3D-CNN + ViT Architecture

---

## 🌟 **Project At A Glance**

This repository documents **Group 7**'s ambitious assignment to classify **8 distinct retinal diseases** from Optical Coherence Tomography (OCT) images. We didn't just train one model—we conducted a head-to-head benchmark of **7 state-of-the-art deep learning architectures**, culminating in the introduction of our own **novel Hybrid 3D-CNN + ViT model**, which achieved **SOTA performance**.

This project is a testament to architectural innovation, rigorous evaluation, and the power of Explainable AI (XAI) in medical imaging.

---

## 🚀 **Key Features & Innovations**

| Feature | Description |
| :--- | :--- |
| **🏆 7 Model Benchmark** | Evaluated InceptionV3, EfficientNet-B0, ResNet18, VGG16, ViT, Swin Transformer, and our Hybrid model. |
| **🧠 Novel Architecture** | Introduced a **Hybrid 3D-CNN + ViT** model that fuses global context (ViT) with local 3D structural features (3D-CNN). |
| **🎯 SOTA Performance** | Our Hybrid model achieved **91.68% test accuracy**, outperforming all others. |
| **👁️ Explainable AI (XAI)** | Integrated **EigenCAM** to visualize *why* the model makes its predictions, providing crucial transparency for medical use. |
| **📊 Rigorous Analysis** | Every model was evaluated with and without data augmentation, revealing nuanced insights into model behavior. |
| **📈 Full Metrics Dashboard** | Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Training Curves, and Model Size for every experiment. |

---

## 🏅 **Performance Leaderboard (Test Set Accuracy)**

Our Hybrid model reigns supreme. Here’s how all models stacked up:

| Rank | Model | Contributor | Basic Accuracy | Augmented Accuracy | Model Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🥇 1st** | **Hybrid 3D-CNN + ViT** | **Soleman Hossain** | **91.68%** | N/A | 328.38 |
| **🥈 2nd** | Vision Transformer (ViT-B/16) | **Soleman Hossain** | 87.89% | 85.96% | 328.43 |
| **🥉 3rd** | ResNet18 | **Sihab Mahmud** | 83.04% | 72.93% | 42.65 |
| **4th** | EfficientNet-B0 | **S. T. A. Mahmud Tonmoy** | 82.18% | 77.46% | 15.33 |
| **5th** | Swin Transformer | **Soleman Hossain** | 82.29% | 79.79% | 105.00 |
| **6th** | VGG16 | **M Shamimul Haque Mondal** | 80.86% | 76.43% | 512.29 |
| **7th** | InceptionV3 | **Soleman Hossain** | 75.07% | 70.07% | 92.95 |

> **Key Insight:** Data augmentation helped weaker models (like InceptionV3) but *hurt* stronger ones (like ViT and ResNet18), suggesting the base dataset is high-quality and augmentation needs careful tuning.

---

## 💡 **The Novelty: Hybrid 3D-CNN + ViT Explained**

Our winning model isn't just an ensemble—it's a **fusion architecture** designed to capture the best of both worlds:

1.  **ViT Branch:** Processes the 2D OCT image to capture **global, contextual features** using self-attention.
2.  **3D-CNN Branch:** Creates a "pseudo-3D volume" from the 2D image (using flips, rotations, etc.) and processes it to capture **local spatial and structural depth features**.
3.  **Fusion Layer:** The features from both branches are concatenated and fed into a final classifier, creating a powerful, multi-perspective representation.

This simple yet effective idea pushed the accuracy from 87.89% (ViT alone) to **91.68%**.

---

## 👥 **Contributors & Responsibilities**

This project was a true team effort. Here’s who was responsible for what:

| Contributor | ID | Model(s) |
| :--- | :--- | :--- |
| **Soleman Hossain** | 2021682042 | InceptionV3 |
| **S. T. A. Mahmud Tonmoy** | 2011105042 | EfficientNet-B0 |
| **Sihab Mahmud** | 21121660642 | ResNet18 |
| **M Shamimul Haque Mondal Shimul** | 2122085642 | VGG16 |
| **Soleman Hossain** | 2021682042 | Vision Transformer (ViT), Swin Transformer, **Hybrid 3D-CNN + ViT (Novelty)**, XAI Integration |

---

## 🛠️ **Technical Stack**

*   **Core Framework:** `PyTorch`
*   **Data Handling:** `torchvision`, `PIL`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Progress Tracking:** `tqdm`
*   **Explainable AI:** `pytorch-grad-cam` (EigenCAM)
*   **Metrics & Analysis:** `scikit-learn`, `numpy`

---

## 📂 **Repository Structure (Conceptual)**

While the code is in a single notebook, it is logically organized by contributor and model:
*   `InceptionV3/` - Soleman Hossain's code and results.
*   `EfficientNet-B0/` - S. T. A. Mahmud Tonmoy's code and results.
*   `ResNet18/` - Sihab Mahmud's code and results.
*   `VGG16/` - M Shamimul Haque Mondal's code and results.
*   `ViT_Swin/` - Transformer models and benchmarking.
*   `Hybrid_Model/` - The novel 3D-CNN + ViT architecture (the crown jewel).
*   `XAI/` - Prediction and EigenCAM visualization code.

---

## 📈 **Why This Project Matters**

This isn't just an academic exercise. It provides a **complete, end-to-end blueprint** for multi-class medical image classification, featuring:

*   A **novel, high-performing architecture** ready for real-world application.
*   A **comprehensive benchmark** to guide future model selection.
*   **Critical insights** into the impact of data augmentation.
*   **Explainability** for building trust in AI-driven medical diagnostics.

It’s a powerful demonstration of how deep learning can be harnessed to tackle complex, real-world medical challenges.
