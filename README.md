# 🍽️ Restaurant Revenue Predictor

A simple Streamlit web app that predicts whether a restaurant will have **High** or **Low** revenue, based on details like location, cuisine, pricing, ratings, and more.

Built using a **Random Forest Classifier** trained in `Restaurant_Revenue_prediction.ipynb`.

---

## 🚀 Demo

Fill in restaurant details in the form → get an instant prediction with confidence score.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `Restaurant_Revenue_prediction.ipynb` | Notebook used to train the model |
| `restaurant_model.pkl` | Trained model *(generated from notebook)* |
| `encoders.pkl` | Encoders for text columns *(generated from notebook)* |

---

## 🛠️ Setup

### 1. Clone this repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate model files
Open `Restaurant_Revenue_prediction.ipynb` and run all cells.

> ⚠️ Before running, add this line above the categorical encoding cell (the variable was missing in the original notebook):
> ```python
> categorical_columns = ["Location", "Cuisine", "Parking Availability"]
> ```

This creates `restaurant_model.pkl` and `encoders.pkl`. Place both files in the project's root folder.

### 4. Run the app locally
```bash
streamlit run app.py
```

Open the link shown in your terminal (usually `http://localhost:8501`).

---

## 🌐 Deploy for Free (Streamlit Community Cloud)

1. Push this project to GitHub (include the `.pkl` files).
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select your repo → set main file to `app.py` → **Deploy**

You'll get a free public URL to share your app.

---

## 🧠 How It Works

- Loads the trained Random Forest model and label encoders
- Takes restaurant details as input (location, price, ratings, marketing budget, etc.)
- Encodes categorical fields the same way as training
- Predicts revenue class: **High** or **Low**
- Shows prediction confidence

---

## 📊 Model Info

- **Algorithm:** Random Forest Classifier
- **Target:** Revenue Class (1 = High, 0 = Low, split by median revenue)
- **Top features:** Average Meal Price, Seating Capacity, Location, Cuisine

---

## 📌 Note

This model predicts a revenue **category** (High/Low), not an exact dollar amount.
