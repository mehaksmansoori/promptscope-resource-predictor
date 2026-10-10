"""Reconstruct the notebook model from its original training feature rows."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

def load_model():
    root = Path(__file__).parent
    path = root / 'model.joblib'
    if path.exists():
        return joblib.load(path)
    artifact = json.loads((root / 'model_config.json').read_text())
    data = pd.read_csv(root / 'training_features.csv')
    cat = ['model']
    num = [c for c in artifact['features'] if c not in cat]
    pre = ColumnTransformer([
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num),
    ])
    base = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=.8, random_state=artifact['seed'], n_jobs=1)
    predictor = Pipeline([('preprocessor', pre), ('regressor', MultiOutputRegressor(base,n_jobs=1))])
    predictor.fit(data[artifact['features']], data[artifact['targets']])
    artifact['predictor'] = predictor
    joblib.dump(artifact, path, compress=3)
    return artifact
