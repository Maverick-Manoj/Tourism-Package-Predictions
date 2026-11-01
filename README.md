# Tourism-Package-Predictions
Predicting customer purchases of tourism packages using machine learning.

## Running the project locally

Use Python 3.11 and run these commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
python -m src.validate_data
python -m pytest -q
streamlit run app.py
```

The application uses the saved model in `models/best_model.joblib`. The dataset is stored in `data/tourism.csv`.
