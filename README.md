# Tourism-Package-Predictions
Predicting customer purchase propensity for the Wellness Tourism Package business objective using machine learning.

This project predicts customer purchase propensity for the Wellness Tourism Package.

The dataset contains a binary target variable, `ProdTaken` (1 = purchased, 0 = not purchased). Since the dataset does not separately identify Wellness Tourism Package purchases, this project assumes that `ProdTaken` represents Wellness Tourism Package purchase status.

The course team has confirmed that this interpretation is acceptable, provided the assumption is documented and the modelling approach and business recommendations remain consistent with it.

The dataset itself does not independently verify the package type. This is a documented modelling assumption and a limitation when interpreting the results.

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
