# 👁️ Retinal OCT Image Classification - 8 Classes
### High-Quality Multi-Class Dataset of OCT Images Across 8 Retinal Conditions

### 🔍 Overview

The **Retinal OCT - 8 Classes** dataset contains **24,000 retinal OCT images** categorized into 8 retinal conditions. This dataset was compiled from multiple reputable sources and is designed to support research in **retinal disease classification** using machine learning models.

---

## 📊 Dataset Structure

The dataset is split into three main subsets: **Train**, **Validation**, and **Test**. Each subset contains 8 subfolders corresponding to the retinal conditions. Images are named consistently, e.g., `drusen_test_1001.jpg`.

| **Subset**      | **Description**                                   |
|-----------------|---------------------------------------------------|
| **Train**       | Contains 18,400 images used for training the model|
| **Validation**  | Contains 2,800 images used for model validation   |
| **Test**        | Contains 2,800 images used for testing model performance |

---

## 📔 Classes

The dataset consists of **8 retinal conditions**:

| **Class**   | **Description**                                | **Train** | **Validation** | **Test** | **Total Images** |
|-------------|------------------------------------------------|-----------|----------------|----------|------------------|
| **AMD**     | Age-related Macular Degeneration                | 2,300     | 350            | 350      | 3,000            |
| **CNV**     | Choroidal Neovascularization                   | 2,300     | 350            | 350      | 3,000            |
| **CSR**     | Central Serous Retinopathy                     | 2,300     | 350            | 350      | 3,000            |
| **DME**     | Diabetic Macular Edema                         | 2,300     | 350            | 350      | 3,000            |
| **DR**      | Diabetic Retinopathy                           | 2,300     | 350            | 350      | 3,000            |
| **DRUSEN**  | Yellow deposits under the retina               | 2,300     | 350            | 350      | 3,000            |
| **MH**      | Macular Hole                                   | 2,300     | 350            | 350      | 3,000            |
| **NORMAL**  | Healthy eyes with no abnormalities             | 2,300     | 350            | 350      | 3,000            |
| **Total**   | -                                              | **18,400**| **2,800**      | **2,800**| **24,000**        |

**Total Images**: 24,000  
**Format**: JPEG  
**Image Dimensions**: Images are not uniform and vary in size.

---

## 📂 Folder Structure

Here’s how the dataset is organized:

| **Subset**      | **Subfolders (Classes)**                        | **Naming Convention**                     |
|-----------------|------------------------------------------------|-------------------------------------------|
| **Train**       | `AMD`, `CNV`, `CSR`, `DME`, `DR`, `DRUSEN`, `MH`, `NORMAL` | e.g., `drusen_train_1001.jpg` |
| **Validation**  | `AMD`, `CNV`, `CSR`, `DME`, `DR`, `DRUSEN`, `MH`, `NORMAL` | e.g., `drusen_val_1001.jpg`   |
| **Test**        | `AMD`, `CNV`, `CSR`, `DME`, `DR`, `DRUSEN`, `MH`, `NORMAL` | e.g., `drusen_test_1001.jpg`  |

---

## 🔄 Preprocessing & Augmentation

Several preprocessing steps were applied to prepare the dataset for model training:
- **Image Augmentation**: Techniques like **cropping**, **padding**, and **horizontal flipping** were used to increase the size of the training set and prevent overfitting.
- **Image Dimensions**: The images in the dataset are of varying sizes, but all are in JPEG format.
- **Data Splitting**: The dataset was split into 75% training, 15% validation, and 15% testing.
- **File Naming**: The dataset maintains consistent file naming, such as `<class>_<subset>_<serial_number>.jpg`

---

## 📝 How to Use the Dataset

1. **Training**: Use the `train` folder to train your model. Each class has its own subfolder.
2. **Validation**: Use the `val` folder for model validation.
3. **Testing**: Use the `test` folder to evaluate your model’s performance.

The dataset is suitable for multi-class classification tasks, particularly in the field of **medical image analysis**.

---

## 📑 Citation

If you use this dataset, please cite it as follows:

**APA Citation**  
Obuli Sai Naren. (2021). *Retinal OCT Image Classification - C8 Dataset* [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DSV/2736749](https://doi.org/10.34740/KAGGLE/DSV/2736749)

---

Feel free to download, analyze, and contribute! 📊💻