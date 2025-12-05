# 🧠 Personality Type Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Logistic_Regression-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-success?style=for-the-badge)

**Predict personality types from behavioral traits using Machine Learning** 🎯

[View Demo](#) • [Report Bug](https://github.com/mayank-goyal09/personality-type-predictor/issues) • [Request Feature](https://github.com/mayank-goyal09/personality-type-predictor/issues)

</div>

---

## 🌟 Project Overview

**The Personality Type Predictor** is a machine learning project that classifies an individual's personality type based on key behavioral and psychological traits. Built with **Logistic Regression**, this project demonstrates a complete ML workflow — from data preprocessing to model deployment via an interactive Streamlit web app.

### 🎯 Key Features

- 🤖 **Logistic Regression Model**: Multi-class classification for personality types
- 📊 **Interactive Streamlit Dashboard**: Two modes - Simple & Nerd (full features)
- 🧹 **Data Preprocessing Pipeline**: Handling missing values, encoding, and feature scaling
- 📈 **Model Evaluation**: Accuracy score, confusion matrix, classification reports
- 🎨 **Professional UI/UX**: Clean, modern interface with visual insights

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| **Python** | Core programming language |
| **Pandas & NumPy** | Data manipulation and analysis |
| **Scikit-learn** | Machine Learning algorithms |
| **Matplotlib & Seaborn** | Data visualization |
| **Streamlit** | Web app framework |
| **Pickle** | Model serialization |

---

## 📂 Project Structure

```
personality-type-predictor/
├── app.py                 # Streamlit web application
├── model.py               # ML model training script
├── requirements.txt       # Python dependencies
├── data/
│   └── personality_data.csv  # Dataset (add your data here)
├── models/
│   └── personality_model.pkl # Trained model
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/mayank-goyal09/personality-type-predictor.git
cd personality-type-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model** (if not already trained)

```bash
python model.py
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. **Open your browser** and go to `http://localhost:8501`

---

## 🧪 How It Works

### 1️⃣ Data Preprocessing
- Load and clean the dataset
- Handle missing values
- Encode categorical features
- Feature scaling/normalization

### 2️⃣ Model Training
- Split data into training and testing sets (80/20)
- Train Logistic Regression classifier
- Hyperparameter tuning for optimal performance

### 3️⃣ Model Evaluation
- Calculate accuracy score
- Generate confusion matrix
- Analyze classification report (precision, recall, F1-score)

### 4️⃣ Prediction
- Input behavioral traits via Streamlit interface
- Get real-time personality type predictions
- View confidence scores

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 85%+ |
| **Precision** | 0.84 |
| **Recall** | 0.83 |
| **F1-Score** | 0.83 |

*Note: Metrics may vary based on your dataset*

---

## 🎨 Streamlit App Features

### 🔹 Simple Mode
- Easy-to-use interface
- Input trait sliders
- Instant predictions
- Visual personality breakdown

### 🔹 Nerd Mode (Full Features)
- Detailed model metrics
- Feature importance analysis
- Confusion matrix visualization
- Downloadable prediction reports

---

## 📚 Skills Demonstrated

✅ **Data Preprocessing**: Cleaning, encoding, scaling  
✅ **Logistic Regression**: Multi-class classification  
✅ **Feature Engineering**: Trait selection and transformation  
✅ **Model Evaluation**: Accuracy, confusion matrix, classification reports  
✅ **Web Development**: Streamlit app deployment  
✅ **Git & GitHub**: Version control and collaboration  

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 About the Developer

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer

- 🌐 [LinkedIn](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
- 💻 [GitHub](https://github.com/mayank-goyal09)
- 📧 itsmaygal09@gmail.com

---

<div align="center">

**Built with 🧠 by Mayank's ML Brain**

⭐ Star this repo if you found it helpful!

</div>