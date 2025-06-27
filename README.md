

# 📈 Stock Price Prediction using LSTM (Long Short-Term Memory)

## 🧠 Overview

This project focuses on predicting the future closing prices of a selected stock using a Recurrent Neural Network (RNN) architecture — specifically, an LSTM model. The LSTM is trained on historical stock price data to capture time-series patterns and generate predictions. The goal is to demonstrate the application of deep learning techniques in financial forecasting.

---

## 🔍 Problem Statement

Can we accurately predict the **next day’s closing price** of a stock based on previous trends using LSTM?

---

## 📊 Dataset

- **Source**: [Yahoo Finance](https://finance.yahoo.com/)
- **Stock**: e.g., Apple (AAPL), Tesla (TSLA), etc.
- **Fields Used**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

---

## 🔧 Tools & Technologies

- **Programming Language**: Python
- **Libraries**:
  - `NumPy`, `Pandas` – Data processing
  - `Matplotlib`, `Seaborn` – Visualization
  - `TensorFlow` / `Keras` – LSTM model
  - `scikit-learn` – Scaling & preprocessing
- **Deployment**: Streamlit / Flask (optional)
- **IDE**: Jupyter Notebook / VSCode

---

## 🧱 Project Structure

├── web_stock_price_predictor.py
├── model/
│   └── Latest_stock_price_model.keras
├── stock_price.ipynb
├── requirements.txt

---

## 📈 Model Architecture

- **Input**: Sequence of past 60 days’ closing prices
- **Layers**:
  - LSTM Layer (50 units)
  - Dropout Layer
  - Dense Layer (1 unit)
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam

---

## 🔄 Workflow

1. Load and clean stock data from Yahoo Finance
2. Normalize data using MinMaxScaler
3. Create time-series training sequences
4. Build and train LSTM model
5. Evaluate performance and visualize predictions
6. (Optional) Deploy model with Streamlit or Flask

---

## 📊 Results

- **RMSE**: 0.33
- Model captures upward/downward trends fairly well.
- Visual comparison shows close alignment between predicted and actual closing prices.

![Predicted vs Actual](screenshots/predicted_vs_actual.png)

---

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run web_stock_price_predictor.py


