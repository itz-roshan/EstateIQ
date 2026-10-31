# 🏡 EstateIQ — AI-Powered Real Estate Intelligence Platform

<p align="center">
  <b>Turning Real Estate Data into Smarter Decisions with Artificial Intelligence</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

<p align="center">
  <a href="https://github.com/itz-roshan/EstateIQ">GitHub Repository</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation--setup">Installation</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

---

## 🌟 Overview

**EstateIQ** is an AI-powered real estate intelligence platform designed to explore property-related data and demonstrate how Machine Learning and Deep Learning can support real estate decision-making.

The platform brings multiple AI-powered modules together in one interactive Streamlit application, covering property price estimation, property image classification, demand forecasting, and customer review sentiment analysis.

Instead of relying only on traditional property information, EstateIQ explores how data-driven techniques can provide useful insights into the real estate market.

## ✨ Features

| Module | Description | Technology |
|---|---|---|
| 🇺🇸 US Home Estimate | Estimates historical home prices using property features | Machine Learning |
| 🇮🇳 India Property Estimate | Explores property prices using locality-based comparable listings and ANN | Python, ANN |
| 🏠 Room Photo Checker | Classifies room images into different room categories | CNN, TensorFlow |
| 📈 Demand Forecast | Forecasts future property demand using historical data | Time Series / Deep Learning |
| 💬 Review Sentiment | Analyzes customer reviews and identifies sentiment | RNN, NLP |
| 📊 Explore Datasets | Helps explore and understand datasets through interactive analysis | Pandas, Streamlit |

## 🧠 AI & Machine Learning Modules

### 1. 🇺🇸 US Home Price Estimation
- Uses historical King County housing data.
- Takes property-related features as input.
- Applies a trained machine learning pipeline to estimate historical property prices.
- Includes renovation-year handling and input validation.

### 2. 🇮🇳 India Property Intelligence
- Explores property prices based on location and property characteristics.
- Uses locality-level comparable listing information.
- Provides indicative price references with confidence limitations.
- Includes an ANN-based academic model.

### 3. 🏠 Property Image Classification
- Uses a Convolutional Neural Network (CNN).
- Analyzes uploaded room images.
- Predicts room categories such as bedroom, living room, and dining room.
- Includes uncertainty handling for low-confidence predictions.

### 4. 📈 Property Demand Forecasting
- Explores historical demand patterns.
- Applies forecasting techniques to understand possible future trends.
- Presents results through visualizations.

### 5. 💬 Customer Review Sentiment Analysis
- Uses Natural Language Processing (NLP).
- Applies a Recurrent Neural Network (RNN).
- Processes customer review text.
- Classifies sentiment into relevant categories.

## 🛠️ Tech Stack

**Programming & Development**
- Python
- Jupyter Notebook
- VS Code

**Machine Learning & Deep Learning**
- TensorFlow / Keras
- Scikit-learn
- ANN
- CNN
- RNN

**Data Analysis**
- Pandas
- NumPy
- Matplotlib
- Seaborn

**Visualization & Application**
- Streamlit
- Plotly

**Tools**
- Git
- GitHub

## 📂 Project Structure

```text
EstateIQ/
│
├── .streamlit/
│
├── Customer_Review_Sentiment_Analysis_RNN/
│
├── Location Based Price Prediction using ANN/
│
├── Property Image Classification Using CNN/
│
├── Property demand forecast/
│
├── Property prize prediction/
│
├── app.py
├── india_comparable.py
│
├── requirements.txt
├── run_app.bat
│
├── validate_india_module.py
├── validate_model_assets.py
├── validate_project.py
│
├── APP_REPAIR_REPORT.md
├── VALIDATION_REPORT.txt
└── README.md
```

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/itz-roshan/EstateIQ.git
```

### Step 2: Navigate to the Project Folder

```bash
cd EstateIQ
```

### Step 3: Create a Virtual Environment

```bash
py -3.12 -m venv .venv
```

### Step 4: Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📊 Project Highlights

- Multiple AI and Deep Learning modules integrated into one application.
- Interactive web interface built using Streamlit.
- Data preprocessing and model-based prediction workflows.
- Visual exploration of real estate datasets.
- Input validation and uncertainty-aware predictions.
- Modular project structure for easier maintenance.

## ⚠️ Important Notes

- US Home Estimate uses historical King County housing data and should not be treated as a current market valuation.
- India Property Estimate provides indicative comparable-based references. Its ANN model has limitations and should not be considered a reliable professional valuation tool.
- Room Photo Checker is an academic image-classification model. Predictions may be uncertain.
- The Review Sentiment module requires its dataset. The large `Reviews.csv` file is not included in this GitHub repository due to its size.
- Forecasting and prediction results depend on the available datasets and model performance.

**EstateIQ is an academic and demonstration project, not a professional property valuation or investment advisory service.**

## 🚀 Future Enhancements

- Integration of live real estate market data.
- Improved prediction accuracy with larger and better-quality datasets.
- Interactive geographical property visualizations.
- More advanced demand forecasting models.
- Enhanced user experience and reporting features.
- Cloud deployment and scalable model serving.

## 👨‍💻 Developer

**Roshan Kumar**

Master of Computer Applications (MCA)  
Invertis University, Bareilly, Uttar Pradesh

Interested in Data Analytics, Machine Learning, Deep Learning, and AI-powered applications.

<p align="center">
  <a href="https://github.com/itz-roshan">
    <img src="https://img.shields.io/badge/GitHub-itz--roshan-181717?style=for-the-badge&logo=github" />
  </a>
</p>

---

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>

<p align="center">
  <b>EstateIQ — Making Real Estate Smarter with AI. 🏡</b>
</p>
