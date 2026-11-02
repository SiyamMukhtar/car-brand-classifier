# Car Brand Classifier

Deep-learning project that recognises the **brand of a car from a photo** (20 brands). A FastAPI endpoint returns the predicted brand and a confidence score, and a small React web app lets you try it in the browser.

## Workflow

| Stage | What was done | Script |
|---|---|---|
| Data collection | Stanford Cars dataset (Hugging Face `tanganke/stanford_cars`), regrouped from 196 car models into the **20 most frequent brands** | `src/download_dataset.py` |
| Data cleaning | Removed corrupt / too-small images (< 150 px) and near-duplicates (perceptual hash), resized to max 512 px, converted to RGB JPEG | `src/clean.py` |
| Data labeling | Label = brand folder name; numeric ids + `class_names.json` | `src/label.py` |
| Preprocessing | Resize 224x224, ImageNet normalisation; augmentation (flip, rotation, colour jitter) on the training set only | `src/dataset.py` |
| Data splitting | Stratified 70 / 15 / 15 split (train 8,677, val 1,859, test 1,861), seed 42 | `src/split.py` |
| Model selection | CNN with transfer learning: **ResNet18** pretrained on ImageNet, new final layer with 20 outputs | `src/model.py` |
| Training | Phase 1: only the new layer (3 epochs); Phase 2: fine-tune all layers (cosine LR). Class-weighted cross-entropy for imbalanced classes. Trained on a Colab T4 GPU. | `src/train.py`, `notebooks/train_colab.ipynb` |
| Evaluation | Test set used once, after training | `src/evaluate.py` |
| Deployment | FastAPI service with a two-stage pipeline (car gate, then brand classifier) and a React web app | `api/main.py`, `frontend/` |

## Dataset

- 12,397 images after cleaning, 20 classes: Acura, Aston Martin, Audi, Bentley, BMW, Buick, Chevrolet, Chrysler, Dodge, Ferrari, Ford, GMC, Honda, Hyundai, Jeep, Lamborghini, Mercedes-Benz, Nissan, Suzuki, Toyota.
- Class imbalance of about 5.7x (Chevrolet 1,757 images vs Lamborghini 308). Handled with a stratified split, class weights and macro-F1 as the main metric.

## Results

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 0.885 | **0.883** |
| Macro-F1 | 0.886 | **0.885** |

Per-brand scores: `outputs/test_report_resnet18.txt`. Confusion matrix: `outputs/confusion_matrix_resnet18.png`. Training curves: `outputs/curves_resnet18.png`.

The most common mistakes are between brands that share body styles, e.g. Dodge -> Chevrolet, GMC <-> Chevrolet, Mercedes-Benz -> BMW.

### Test with new internet photos (not from the dataset)

8 of 10 car photos were classified correctly; the cat photo was rejected by the car gate. Full table: `outputs/api_test_results.md`.

## How a photo is processed

1. **Car gate:** an ImageNet-pretrained ResNet18 sums the probability of all car-related classes. If this is below 0.20 the API answers `"is_car": false` and no brand is predicted (cat, food, people, ...). The threshold was chosen on our own photos: a cat scored 0.00, real cars scored 0.89 to 1.00, and 95% of 150 test-set cars scored above 0.88.
2. **Brand classifier:** our own trained ResNet18 predicts one of the 20 brands with a confidence score.

## API

```bash
pip install -r requirements.txt
uvicorn api.main:app
```

Open `http://127.0.0.1:8000/docs` for the interactive page, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@my_car.jpg"
```

Example response:

```json
{
  "is_car": true,
  "car_probability": 0.997,
  "predicted_class": "BMW",
  "confidence": 0.996,
  "top3": [
    {"class": "BMW", "confidence": 0.996},
    {"class": "Audi", "confidence": 0.002},
    {"class": "Mercedes-Benz", "confidence": 0.001}
  ]
}
```

`GET /health` checks that the models are loaded. If the confidence is below 0.5 the response also contains a `warning` field.

## Web app

```bash
uvicorn api.main:app          # terminal 1
cd frontend
npm install
npm run dev                   # terminal 2, then open http://localhost:5173
```

## Reproduce the project

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python src/download_dataset.py
python src/clean.py
python src/label.py
python src/split.py
python src/train.py               # a GPU is recommended; see notebooks/train_colab.ipynb
python src/evaluate.py
uvicorn api.main:app
```

The trained weights (`models/resnet18_best.pth`, about 44 MB) are included in the repository.

## Limitations

- **Closed-set classifier:** the brand model only knows these 20 brands. The car gate rejects images that are not cars, but a real car of another brand (e.g. Tesla) passes the gate and is still assigned to one of the 20 brands, sometimes with high confidence. A future version should add an "other brand" class.
- The train / validation / test split is random per image, so the same car models appear (in different photos) in all three sets. Performance on completely new car models may be lower.
- The dataset consists mostly of cars up to 2012; newer models may be recognised less reliably.
- The Hugging Face dataset page does not state a licence. This project is for educational purposes.

## Project structure

```
CarClassifier/
├── src/         data pipeline, model, training, evaluation
├── api/         FastAPI service
├── frontend/    React web app
├── tests/       scripts that test the API and the car gate
├── models/      trained weights
├── outputs/     plots, reports, test results
├── notebooks/   Colab training notebook
├── data/        splits.csv, class_names.json, labels.csv (images are not stored in git)
└── requirements.txt
```