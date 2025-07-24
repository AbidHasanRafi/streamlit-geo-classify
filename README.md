# PatternNet Land Use Classification App

A lightweight CNN-powered web application built with **Streamlit** for classifying remote sensing images into **7 generalized land-use categories** from the **PatternNet** dataset. This app is ideal for fast, resource-efficient deployment on CPUs or limited hardware.

## Demo

Upload an aerial image (e.g., from PatternNet), and classify it into categories like **Transport**, **Urban**, **Water**, etc., using a trained lightweight CNN.

![App Demo](https://raw.githubusercontent.com/AbidHasanRafi/streamlit-geo-classify/main/app.png) 

---

## Generalized Class Groupings

| Group          | Original PatternNet Classes                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Transport**  | airplane, bridge, freeway, railway, runway, runway\_marking, shipping\_yard                                                |
| **Urban**      | dense\_residential, sparse\_residential, mobile\_home\_park, parking\_lot, parking\_space, nursing\_home, coastal\_mansion |
| **Sports**     | baseball\_field, basketball\_court, football\_field, tennis\_court                                                         |
| **Water**      | beach, ferry\_terminal, harbor, river                                                                                      |
| **Industrial** | oil\_well, oil\_gas\_field, shipping\_yard, storage\_tank, wastewater\_treatment\_plant                                    |
| **Vegetation** | cemetery, chaparral, christmas\_tree\_farm, forest, closed\_road                                                           |
| **Other**      | crosswalk, golf\_course, intersection, solar\_panel, swimming\_pool, transformer\_station                                  |

---

## Model

A custom **lightweight CNN** (4 convolutional layers with batch norm, pooling, and adaptive avg-pooling) trained on grouped classes from the PatternNet dataset.

**Key Features:**

* Lightweight model (optimized for CPU inference)
* 7-way classification
* Softmax output with probability visualization
* Real-time prediction from uploaded image

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AbidHasanRafi/streamlit-geo-classify.git
cd streamlit-geo-classify
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

**Required Libraries:**

* `torch`
* `torchvision`
* `streamlit`
* `Pillow`
* `numpy`

### 3. Download Model

Place your trained model file in the root directory:

```bash
best_model.pth
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Usage

1. Open your browser at `http://localhost:8501`.
2. Upload an image (`.jpg`, `.jpeg`, `.png`) from the PatternNet dataset or your own aerial image.
3. Click **"Classify Image"**.
4. View:

   * Predicted class
   * Class probabilities
   * Bar chart visualization

---

## Dataset

You can download sample images from the PatternNet dataset:

[![Kaggle](https://img.shields.io/badge/Kaggle-PatternNet_Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/abidhasanrafi/patternnet)