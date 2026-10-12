# GlucoBench Glucose Prediction

This version implements:

1. Data cleaning
2. Data preprocessing
3. KNN imputation
4. Glucose regression

## Dataset
`dataset/GlucoBench_benchmark_dataset.csv`

Target variable: `glucose`

## Data cleaning
- Missing values are identified before modeling.
- The `notes` column is completely empty and is removed.
- `user_id` and `device_id` are identifiers and are removed from the model.
- `timestamp` is converted into useful time features.
- `glucose_roll_mean_1h` is removed because it contains the current glucose value and would leak the target.

## KNN imputation
Numeric missing values are imputed inside the sklearn pipeline using:

- StandardScaler
- KNNImputer
- 5 nearest neighbors
- distance-based weighting

The imputer is fitted only on training data, preventing train/test leakage.

## Categorical preprocessing
Categorical columns are:
- most-frequent imputed
- one-hot encoded
- unknown categories are handled safely

## Regression
The target is `glucose`.

A chronological 80/20 train-test split is used. The regression model is Ridge Regression.

Current dataset result:
- MAE: approximately 0.51 mg/dL
- RMSE: approximately 0.67 mg/dL
- R²: approximately 0.9996

The exact metrics are recalculated when `/regression` is opened.

## Run

Create/activate your virtual environment and install:

```bash
pip install -r requirements.txt
```

Then:

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

Pages:
- `/load` - dataset preview
- `/eda` - exploratory data analysis
- `/preprocessing` - cleaning and KNN imputation explanation
- `/regression` - train and evaluate the regression model
