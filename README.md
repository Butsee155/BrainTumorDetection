# 🧠 Brain Tumor Detection System

> An AI-powered desktop application that uses a CNN deep learning model (EfficientNetB0) to detect and classify brain tumors from MRI scans — with Grad-CAM heatmap visualization, tumor region annotation, batch processing, and a full admin dashboard.

---

> ⚠️ **Disclaimer:** This system is developed for academic, research, and educational purposes only. It is not a certified medical device and must not be used for clinical diagnosis. Always consult a qualified radiologist or neurosurgeon.

---

## 📌 Overview

The **Brain Tumor Detection System** allows medical professionals and researchers to upload MRI brain scans for automated analysis. The system classifies the scan into one of four categories — Glioma, Meningioma, Pituitary Tumor, or No Tumor — and provides a confidence score, Grad-CAM heatmap overlay, tumor region bounding box, severity level, recommendation, and specialist referral. All results are saved to a SQL Server database with full export capabilities.

---

## ✨ Features

- 🧠 **CNN Classification** — EfficientNetB0 transfer learning model (4 classes)
- 🔥 **Grad-CAM Heatmap** — Visual overlay showing which regions activated the prediction
- 📍 **Tumor Region Annotation** — Bounding box drawn around detected tumor area
- 📊 **Top 3 Predictions** — Each with confidence score and visual progress bar
- ⚠️ **Severity Levels** — Low / Medium / High / Critical with colour coding
- ✅ **Recommendations** — Specific action advice per tumor type
- 👨‍⚕️ **Specialist Referral** — Recommended doctor type per diagnosis
- 📦 **Batch Processing** — Analyse multiple MRI images at once
- 💾 **SQL Server** — All scans saved automatically with patient details
- 📊 **Admin Dashboard** — Stats per tumor type, full scan history
- 📤 **Export Reports** — Excel and CSV export for all scans, today, critical cases

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x + Keras |
| CNN Architecture | EfficientNetB0 (Transfer Learning + Fine-tuning) |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Image Processing | OpenCV, Pillow, NumPy |
| GUI Framework | Tkinter (Modern Dark theme) |
| Database | Microsoft SQL Server 2019 + SSMS 20 |
| DB Connector | pyodbc |
| Export | openpyxl, csv |

---

## 📁 Project Structure

```
BrainTumorDetection/
│
├── main_app.py              # Login screen & launcher
├── detector.py              # Single MRI upload + analysis GUI
├── batch_detector.py        # Batch MRI upload + analysis GUI
├── dashboard.py             # Admin dashboard (stats, history, export)
├── model_engine.py          # CNN prediction + Grad-CAM + DB logging
├── train_model.py           # One-time model training script
├── db_config.py             # SQL Server connection
├── models/
│   └── brain_tumor_model.h5 # Trained CNN model
├── results/                 # Auto-saved heatmaps and annotated images
└── data/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── notumor/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── notumor/
```

---

## 🗄️ Database Schema

```sql
CREATE DATABASE BrainTumorDetection;

CREATE TABLE ScanHistory (
    ScanID         INT IDENTITY(1,1) PRIMARY KEY,
    PatientName    VARCHAR(200),
    PatientAge     INT,
    PatientGender  VARCHAR(20),
    ImagePath      VARCHAR(500),
    HeatmapPath    VARCHAR(500),
    Prediction1    VARCHAR(100),
    Confidence1    FLOAT,
    Prediction2    VARCHAR(100),
    Confidence2    FLOAT,
    Prediction3    VARCHAR(100),
    Confidence3    FLOAT,
    FinalDiagnosis VARCHAR(100),
    SeverityLevel  VARCHAR(50),
    Recommendation VARCHAR(MAX),
    ScannedAt      DATETIME DEFAULT GETDATE()
);

CREATE TABLE TumorInfo (
    TumorID        INT IDENTITY(1,1) PRIMARY KEY,
    TumorType      VARCHAR(100),
    Description    VARCHAR(MAX),
    SeverityLevel  VARCHAR(50),
    Recommendation VARCHAR(MAX),
    Specialist     VARCHAR(200),
    Urgency        VARCHAR(100)
);
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 — [python.org](https://python.org)
- SQL Server 2019 Express — [Microsoft](https://www.microsoft.com/en-us/sql-server)
- SSMS 20 — [Microsoft](https://learn.microsoft.com/en-us/sql/ssms)
- ODBC Driver 17 for SQL Server

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/BrainTumorDetection.git
cd BrainTumorDetection
```

### 3. Install Dependencies
```bash
py -3.10 -m pip install tensorflow opencv-python numpy pillow pyodbc openpyxl scikit-learn matplotlib
```

### 4. Download Dataset
Download the Brain Tumor MRI Dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract into the `data/` folder.

### 5. Set Up Database
Run the SQL script in SSMS to create the database, tables, and seed tumor reference data.

### 6. Train the Model (Run Once — ~20–40 mins)
```bash
py -3.10 train_model.py
```

### 7. Launch the Application
```bash
py -3.10 main_app.py
```

> Default password: `admin123`
> Change in `main_app.py` → `ADMIN_PASSWORD = "your_password"`

---

## 🚀 How It Works

1. **Doctor uploads MRI scan** — single image or batch of images
2. **CNN preprocessing** — image resized to 224×224, normalised
3. **EfficientNetB0 prediction** — returns top 3 classes with probability scores
4. **Grad-CAM heatmap** — gradient-weighted activation map overlaid on original scan
5. **Tumor region annotation** — contour detection draws bounding box around activation region
6. **DB lookup** — severity, recommendation, and specialist fetched from SQL Server
7. **Results displayed** — confidence bars, severity badge, recommendation, specialist
8. **Auto-logged** — every scan saved to `ScanHistory` with full patient details

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| `AttributeError: layer sequential has never been called` | Added model warm-up with dummy input immediately after loading to force all layer outputs to be defined |
| Grad-CAM failing on EfficientNet + Sequential wrapper | Built 3-level fallback system — Method 1 (base model layers) → Method 2 (GAP input) → fallback colour heatmap |
| Testing dataset folder empty causing `ValueError: PyDataset length 0` | Fixed dataset path to match extracted Kaggle folder structure |
| Tkinter `TclError: expected integer but got "7.5"` | Replaced all float font sizes with integers throughout all GUI files |
| Clock callback crashing after window destroy | Wrapped `update_clock` in `try/except tk.TclError` to stop cleanly |
| Lambda variable conflict with exception variable `e` | Renamed exception variables to `analysis_err`, `batch_err` and extracted message string before lambda capture |

---

## 📸 Screenshots

> _Add your screenshots here_

| Login Screen | MRI Analyser |
|---|---|
| ![Login](screenshots/login.png) | ![Analyser](screenshots/analyser.png) |

| Grad-CAM Heatmap | Admin Dashboard |
|---|---|
| ![Heatmap](screenshots/heatmap.png) | ![Dashboard](screenshots/dashboard.png) |

| Batch Analyser | Export Reports |
|---|---|
| ![Batch](screenshots/batch.png) | ![Export](screenshots/export.png) |

---

## 🔮 Future Improvements

- [ ] Larger dataset and additional tumor subtypes
- [ ] 3D MRI volume analysis
- [ ] DICOM medical image format support
- [ ] Web-based version using Flask
- [ ] Patient profile management system
- [ ] PDF report generation per scan
- [ ] Integration with hospital management systems

---

## 👤 Author

**R.M. Nisitha Nethsilu**
🔗 [LinkedIn](https://linkedin.com/in/nisithanethsilu)
🐙 [GitHub](https://github.com/Butsee155)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project useful, please give it a star on GitHub!
