House Segmentation Service

This repository upgrades the original Lab 1 containerized API into a Lab 2 project for aerial house segmentation. It now includes:

- secrets injection with `.env` and `python-dotenv`
- a Flask API that serves segmentation predictions
- a PyTorch U-Net baseline
- dataset preparation, training, and evaluation scripts
- automated tests
- a GitHub Actions CI/CD pipeline for testing, building, and optional Docker Hub publishing

Project Structure

- `app.py`: Flask API entry point
- `model_service/`: segmentation model, config, metrics, inference, and training helpers
- `scripts/prepare_dataset.py`: creates train/val/test splits and binary masks
- `scripts/train_segmentation.py`: trains the U-Net and saves checkpoints/curves
- `scripts/evaluate_segmentation.py`: computes IoU and Dice on the test split and saves prediction panels
- `tests/`: API and metric tests
- `.github/workflows/ci-cd.yml`: CI/CD workflow

Secrets Injection

1. Copy `.env.example` to `.env`.
2. Update the values, especially `MODEL_SERVICE_API_KEY` if you want the API protected.
3. The app loads environment variables dynamically through `Settings.from_env()`.

Example `.env` usage:

```bash
cp .env.example .env
```

If `REQUIRE_API_KEY=true`, send requests with the `X-API-Key` header.

Dataset Preparation

There are two supported dataset-prep modes:

1. `local`
   Use your own aerial images plus existing masks or polygon annotations.
2. `week7_sam`
   Reproduce the Week 7 notebook workflow: load the satellite-building dataset, run SAM, compare SAM masks against building bounding boxes with IoU, and save the accepted pseudo-label masks.

The local mode supports:

- binary or color mask images
- LabelMe-style JSON polygons
- simple polygon JSON exports

Local example:

```bash
python scripts/prepare_dataset.py \
  --source local \
  --images-dir data/raw/images \
  --annotations-dir data/raw/annotations \
  --output-dir data/processed
```

If your Week 7 annotation export uses a different JSON schema, update `build_mask_from_json()` inside `scripts/prepare_dataset.py`.

Week 7 SAM example:

1. Install the extra dependencies:

```bash
pip install -r requirements-sam.txt
```

2. Download the SAM checkpoint, for example `sam_vit_h_4b8939.pth`.

3. Run:

```bash
python scripts/prepare_dataset.py \
  --source week7_sam \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --sam-device cpu \
  --hf-splits train \
  --limit 3 \
  --sam-points-per-side 16 \
  --sam-points-per-batch 32 \
  --verbose \
  --match-iou-threshold 0.3 \
  --output-dir data/processed
```

That command will:

- load the Week 7 Hugging Face dataset
- generate candidate masks with SAM
- keep SAM masks that agree with labeled building boxes
- fall back to the box mask when no SAM region matches
- save `train/val/test` image and mask folders

For a quick smoke test, keep `--limit 10` or `--limit 20`. Once it works, remove the limit for the full run.
For a faster smoke test on CPU, use `vit_b`, `--limit 1` or `--limit 3`, and lower `--sam-points-per-side` to `16`.

Training

```bash
python scripts/train_segmentation.py \
  --data-dir data/processed \
  --output-dir artifacts \
  --epochs 10 \
  --batch-size 4 \
  --image-size 256 \
  --device cpu
```

Training outputs:

- `artifacts/checkpoints/best_model.pt`
- `artifacts/training/training_history.json`
- `artifacts/training/training_curves.png`

Evaluation

```bash
python scripts/evaluate_segmentation.py \
  --data-dir data/processed \
  --weights artifacts/checkpoints/best_model.pt \
  --output-dir artifacts/evaluation \
  --image-size 256
```

Evaluation outputs:

- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/prediction_*.png`

API Usage

Run locally:

```bash
python app.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Prediction request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: replace-with-a-secret-key" \
  -F "file=@data/raw/images/sample.png"
```

You can optionally include a ground-truth mask in the same request to get per-image IoU and Dice:

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: replace-with-a-secret-key" \
  -F "file=@data/raw/images/sample.png" \
  -F "ground_truth=@data/raw/annotations/sample.png"
```

Docker

Build:

```bash
docker build -t house-segmentation-service .
```

Run:

```bash
docker run --env-file .env -p 5000:5000 house-segmentation-service
```

If your trained checkpoint is stored on your machine rather than baked into the image, mount it into the container:

```bash
docker run --env-file .env \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -p 5000:5000 \
  house-segmentation-service
```

Tests

```bash
pytest
```

CI/CD

The workflow in `.github/workflows/ci-cd.yml` does three things:

- runs the test suite
- builds the Docker image
- pushes the image to Docker Hub when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are configured in GitHub Actions secrets

Suggested Report Content

- dataset description and how masks were generated from the Week 7 SAM + IoU workflow
- model architecture and training setup
- IoU and Dice results on the test split
- screenshots of GitHub Actions runs and saved prediction panels
- challenges such as overfitting, class imbalance, or annotation noise
