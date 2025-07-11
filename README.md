# 📦 MLCrafter

MLCrafter is an interactive terminal-based tool designed to help you create, train, and use custom **Linear Regression** models on your own data — without writing a single line of ML code! It supports CSV import, manual data entry, automatic preprocessing (including encoding for categorical variables), and model saving for later prediction or visualization.

---

## ✨ Features

* 📊 Create custom datasets manually
* 🧠 Train linear regression models using `scikit-learn`
* 🔁 Automatically handle categorical inputs via `get_dummies`
* 📈 Plot actual vs predicted data
* 💾 Save and reuse trained models
* 🔮 Predict results using dynamic input from user

---

## 📁 Folder Structure

```
MLCrafter/
├── dataframes/           # CSVs created using manual data entry
├── trained_models/       # Saved ML models and metadata
├── mlcrafter.py          # Main script to run
├── README.md             # You're here
└── requirements.txt      # Python dependencies
```

---

## 🔧 Requirements

Install dependencies with pip:

```
pip install -r requirements.txt
```

### `requirements.txt` content:

```
pandas
numpy
matplotlib
scikit-learn
joblib
```

---

## 🚀 How to Use

Run the tool using:

```bash
python mlcrafter.py
```

You'll be prompted with the following options:

### 🔹 Menu

```
1 - Create DF for ML Model
2 - Train Your Own ML Model
3 - Use Existing Models
4 - Exit
```

---

## 🧪 Workflow Example

1. **Create Dataset**

   * Choose option `1`
   * Manually enter column names and rows
   * CSV saved in `dataframes/`

2. **Train Model**

   * Choose option `2`
   * Select CSV file
   * Pick input/output columns
   * Tool auto-handles categorical encoding
   * Model trained and evaluated (R² score, MAE, RMSE)
   * Model saved in `trained_models/`

3. **Use Model**

   * Choose option `3`
   * Load saved model from list
   * You can:

     * 🔍 View prediction graph using any input column
     * ✍️ Enter new data (even with strings) for live predictions

---

## 🧠 Model Handling

* Models are saved using `joblib` as `.pkl` files
* Metadata (input columns, dummy columns, etc.) is bundled
* User prediction input auto-encoded to match model training
* No need to worry about strings or manual transformation

---

## 🤖 Author

👤 Made by **TheGhostLoop** — a curious mind learning and crafting ML tools.

---

## 📜 License

MIT License — feel free to fork, modify, and share!
