# FoodIntel Model Training

Default backbone: `mobilenet_v3_small`

Supported models:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `resnet18`

Custom ImageFolder training:

```bash
python ml/train.py \
  --dataset-source imagefolder \
  --data-dir ml/dataset \
  --model mobilenet_v3_small \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.0003 \
  --output ml/models/food_model.pt \
  --classes-output ml/classes.json \
  --reports-dir ml/reports
```

Food-101 baseline training:

```bash
python ml/train.py \
  --dataset-source food101 \
  --food101-root ml/raw_data \
  --selected-classes pizza,hamburger,omelette \
  --model mobilenet_v3_small \
  --epochs 5 \
  --batch-size 32 \
  --output ml/models/food101_model.pt
```

Evaluation:

```bash
python ml/evaluate.py --data-dir ml/dataset --model-path ml/models/food_model.pt --classes-path ml/classes.json
```

Local prediction:

```bash
python ml/predict_local.py --image sample.jpg --model-path ml/models/food_model.pt --classes-path ml/classes.json
```

Training outputs:

- best checkpoint in `ml/models/`
- classes map in `ml/classes.json`
- classification report and metrics in `ml/reports/`
- confusion matrix image in `ml/reports/confusion_matrix.png`

Do not report fake accuracy values. Final performance depends heavily on the dataset quality and class balance.

Recommended FoodIntel workflow:

1. Use Food-101 only as a pipeline sanity check.
2. Prepare a Nigerian dataset that maps into FoodIntel class slugs.
3. Retrain the final model on those Nigerian classes.
4. Regenerate `ml/classes.json` from that dataset so backend predictions align with food slugs and nutrition records.

Clean the extensive dataset before v2 training:

```bash
python ml/build_clean_dataset.py \
  --input-dir ml/dataset_extensive \
  --output-dir ml/dataset_extensive_clean \
  --min-total-images 150 \
  --clear-output
```

Then train the cleaner dataset:

```bash
python ml/train.py \
  --dataset-source imagefolder \
  --data-dir ml/dataset_extensive_clean \
  --model mobilenet_v3_small \
  --epochs 12 \
  --batch-size 24 \
  --lr 0.0002 \
  --image-size 160 \
  --num-workers 4 \
  --early-stop-patience 3 \
  --output ml/models/food_model_extensive_v2.pt \
  --classes-output ml/classes.json \
  --reports-dir ml/reports
```

Safe feedback loop for continuous improvement:

1. Users submit prediction feedback through the backend feedback endpoint.
2. Feedback stays in a review queue and is **not** used for automatic self-training.
3. Only approved feedback samples are exported into a retraining dataset.
4. Retrain offline, evaluate, and promote the new checkpoint only if metrics improve.

Export approved feedback images:

```bash
python ml/export_feedback_dataset.py \
  --mongodb-uri "$MONGODB_URI" \
  --db-name foodintel_db \
  --output-dir ml/dataset_feedback \
  --uploads-dir backend/uploads \
  --status approved \
  --clear-output
```
