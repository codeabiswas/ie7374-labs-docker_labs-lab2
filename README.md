# Heart Disease Prediction

## What Changed

- **Dataset**: Replaced Iris (toy classification) with the [UCI Heart Disease (Cleveland)](https://openml.org/d/53) dataset — 303 patient records, 13 clinical features, binary outcome (disease / no disease)
- **Model**: Replaced the TensorFlow neural network with a scikit-learn `RandomForestClassifier` inside a `Pipeline` (with `StandardScaler`) — lighter, no GPU needed, faster to train
- **Serialization**: Switched from `.keras` format to `joblib` (`.joblib` files) — standard for sklearn models; both model and feature names are saved separately
- **New endpoint**: Added `GET /feature-importance` — returns a JSON array of features ranked by their Random Forest importance score
- **UI**: Replaced 4 flower-measurement inputs with a 2-column form of 13 clinical inputs (numeric fields and dropdowns); result displays a confidence bar and heart emoji instead of a flower image
- **Dependencies**: Removed `tensorflow` and `keras`; added `pandas` for data loading

## What This Project Does

- Trains a Random Forest classifier on clinical patient data to predict heart disease risk
- Serves predictions via a Flask web API on port 4000
- Provides a cyberpunk-styled web form at `/predict` where users enter 13 clinical measurements and receive an instant risk assessment with confidence score
- Exposes `/feature-importance` (JSON) showing which clinical features most influence the model
- Demonstrates Docker multi-stage builds and Docker Compose with shared volumes — two containerization patterns for ML training + serving workflows

## Running Locally

### Option A — Docker Compose (two separate services)

```bash
docker compose up
```

- The `model-training` service trains the model and writes artifacts to a shared volume
- The `serving` service waits for training to complete, then starts the Flask server
- Visit: `http://localhost:80/predict`
- Feature importance: `http://localhost:80/feature-importance`

### Option B — Single Docker image (multi-stage build)

```bash
docker build -t cardioscan .
docker run -p 4000:4000 cardioscan
```

- Visit: `http://localhost:4000/predict`
- Feature importance: `http://localhost:4000/feature-importance`

### Option C — Run directly with Python

```bash
pip install -r requirements.txt
cd src
python model_training.py   # trains and saves model.joblib + feature_names.joblib
python main.py             # starts Flask on port 4000
```

- Visit: `http://localhost:4000/predict`
