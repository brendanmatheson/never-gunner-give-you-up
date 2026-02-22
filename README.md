# ⚽ Match Predictor

Machine learning model to predict Premier League match outcomes with 52.5% accuracy.

**[Try the Live App →](https://brendanmatheson-never-gunner-give-you-up.streamlit.app)**

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.54.0-red)

---

## 📊 Model Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 52.5% | Baseline: 46% (always predict home win) |
| **Brier Score** | 0.197 | Professional models: ~0.21 |
| **Training Data** | 1,466 matches | 2019-2024 EPL seasons |
| **Test Data** | 373 matches | 2023/24 season |

**Note:** 52.5% accuracy is excellent for football prediction. The theoretical ceiling is approximately 55% due to the inherent randomness in the sport.

---

## 🎯 Features

- **Interactive predictions** for any EPL matchup
- **Probability visualization** with interactive charts
- **Form analysis** based on last 5 home/away matches
- **Clean, professional UI** built with Streamlit

---

## 🧠 How It Works

The model uses **Logistic Regression** with features engineered from:

- **Recent Form:** Points from last 5 home/away matches
- **Goal Patterns:** Average goals scored and conceded
- **Form Differentials:** Relative strength between teams
- **Home Advantage:** Implicit in home/away performance splits

### Why Logistic Regression?

After testing multiple models:
- ✅ Logistic Regression: Brier 0.197
- ❌ Random Forest: Brier 0.201
- ✅ Better probability calibration
- ✅ Faster inference
- ✅ More interpretable

Simple often beats complex in ML!

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/brendanmatheson/never-gunner-give-you-up.git
cd never-gunner-give-you-up

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 🎓 Methodology

### Data Collection
- **Source:** [football-data.co.uk](https://www.football-data.co.uk)
- **Timeframe:** 5 EPL seasons (2019-2024)
- **Total matches:** 1,839 (1,466 training, 373 testing)

### Feature Engineering

Key features:
- Recent form (last 5 home/away matches)
- Goals scored/conceded patterns
- Form differentials between teams

### Model Training
- **Algorithm:** Logistic Regression with StandardScaler
- **Validation:** Time-series split (no data leakage)
- **Results:** 52.5% accuracy, 0.197 Brier score


---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn** - Model training
- **pandas/NumPy** - Data processing
- **Streamlit** - Web interface
- **Plotly** - Visualizations

---

## 📜 Data Attribution

Data sourced from [football-data.co.uk](https://www.football-data.co.uk) - free for educational and research purposes.

---

## ⚠️ Disclaimer

This model is for **educational purposes only**. Football matches are inherently unpredictable.

---

## 👤 Author

**Brendan Matheson**
- GitHub: [@brendanmatheson](https://github.com/brendanmatheson)
- LinkedIn: [Brendan Matheson](https://linkedin.com/in/brendan-matheson)

---

⭐ **If you found this project helpful, please star it!**
