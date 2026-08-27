# Spaceship Titanic — SYS-304 Milestone 1: Prototype

## Problem

In the year 2912, the *Spaceship Titanic* collided with a spacetime anomaly while
transporting ~13,000 passengers to three newly habitable exoplanets. Roughly half
the passengers were "transported" to an alternate dimension as a result.

**Task**: Binary tabular classification — given a passenger's demographic, cabin,
and onboard-spending record, predict whether they were `Transported` (`True`/`False`).

## Dataset

[Kaggle: Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

- `train.csv` — 8,693 passengers with labels
- `test.csv` — 4,277 passengers to predict (Kaggle holdout)
- `sample_submission.csv` — expected submission format

Key features: `HomePlanet`, `CryoSleep`, `Cabin` (`Deck/Num/Side`), `Destination`,
`Age`, `VIP`, five onboard-spending columns (`RoomService`, `FoodCourt`,
`ShoppingMall`, `Spa`, `VRDeck`), and `PassengerId` (encodes travel group).

## Approach

1. **EDA** (`SYS-304.ipynb`) — missing-value audit, target balance, distributions
   of numeric/categorical features, correlation heatmap.
2. **Feature engineering** — split `Cabin` into `Deck`/`CabinNum`/`Side`; derive
   `Group`/`GroupSize` from `PassengerId`; sum spending columns into `TotalSpend`.
3. **Preprocessing** — `ColumnTransformer` pipeline: median imputation + scaling
   for numeric features, most-frequent imputation + one-hot encoding for
   categorical features.
4. **Baseline models** — `RandomForestClassifier` and `XGBClassifier`, compared
   on an 80/20 stratified train/validation split.
5. **Evaluation** — accuracy, precision/recall/F1, confusion matrix.

## Results

| Model | Validation Accuracy |
|---|---|
| Random Forest | 80.9% |
| **XGBoost (selected)** | **81.4%** |

This is a naive proof-of-concept baseline (no hyperparameter tuning), intended
as the starting point for the deployment/scaling milestones later in the course.

## Repository Contents

- `SYS-304.ipynb` — full data pipeline, EDA, training, and evaluation
- `baseline_model.pkl` — saved best-performing pipeline (preprocessing + XGBoost)
- `submission.csv` — Kaggle-format predictions on `test.csv`
- `train.csv`, `test.csv`, `sample_submission.csv` — Kaggle source data
- `requirements.txt` — Python dependencies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter notebook SYS-304.ipynb
```
