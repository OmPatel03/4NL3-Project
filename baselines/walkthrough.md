## Data Split

| Split | Range | Count |
|-------|-------|-------|
| Train | Inception → Spy | 701 |
| Val | Lilo & Stitch → American Gangster | 150 |
| Test | Hook → The Giver | 150 |

## Results (Validation Set)

| Baseline | Accuracy |
|----------|----------|
| Random | 10.00% |
| **Majority** (Action) | **22.67%** |
| CBOW + LogReg (counts) | 34.67% |
| **CBOW + LogReg (TF-IDF)** | **40.00%** |


## How to Run

```bash
# 1. Split data (produces baselines/splits/{train,val,test}.json)
python baselines/split_data.py

# 2. Activate venv (scikit-learn is required for CBOW)
source baselines/.venv/bin/activate

# 3. Simple baselines (Random + Majority)
python baselines/simple_baselines.py \
  --train baselines/splits/train.json \
  --eval  baselines/splits/val.json \
  --output baselines/outputs/simple_val_predictions.json

# 4. CBOW + Logistic Regression
python baselines/cbow_baseline.py \
  --train baselines/splits/train.json \
  --eval  baselines/splits/val.json \
  --output baselines/outputs/cbow_val_predictions.json

# 5. CBOW with TF-IDF (recommended)
python baselines/cbow_baseline.py \
  --train baselines/splits/train.json \
  --eval  baselines/splits/val.json \
  --output baselines/outputs/cbow_tfidf_val_predictions.json \
  --use-tfidf
```
