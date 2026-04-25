# FoodIntel Dataset Guide

Recommended paths:

- Best final-year path: train on Nigerian/local food images where possible.
- Fast baseline path: use Food-101 through torchvision.

Direct dataset references:

- Food-101 official page: <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>
- Food-101 torchvision docs: <https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html>
- Food-101 TensorFlow datasets page: <https://www.tensorflow.org/datasets/catalog/food101>
- Kaggle NigerianFoodDataset: <https://www.kaggle.com/datasets/hakymulla/nigerianfooddataset>
- Kaggle Nigerian foods and snacks multi-class: <https://www.kaggle.com/datasets/peaceedogun/nigerian-foods-and-snacks-multiclass>
- Roboflow Nigeria food dataset: <https://universe.roboflow.com/nigeria-food/nigeria-food/dataset/6>
- USDA FoodData Central: <https://fdc.nal.usda.gov/>
- Open Food Facts data: <https://world.openfoodfacts.org/data>

Expected custom dataset structure:

- `ml/dataset/train/jollof_rice/*.jpg`
- `ml/dataset/val/jollof_rice/*.jpg`
- `ml/dataset/test/jollof_rice/*.jpg`

Recommended Nigerian training path:

1. Download one of the Kaggle Nigerian food datasets manually or with Kaggle CLI.
2. Extract it into a raw folder such as `ml/raw_nigerian_foods/`.
3. Create a class mapping JSON based on `ml/nigerian_class_map.example.json`.
4. Convert the raw folders into FoodIntel classes with:

```bash
.venv/bin/python ml/prepare_nigerian_dataset.py \
  --input-dir ml/raw_nigerian_foods \
  --output-dir ml/dataset \
  --class-map ml/nigerian_class_map.example.json \
  --val-split 0.15 \
  --test-split 0.15 \
  --clear-output
```

Example FoodIntel target slugs:

- `jollof_rice`
- `fried_rice`
- `egusi_soup`
- `eba`
- `pounded_yam`
- `beans`
- `moi_moi`
- `akara`
- `plantain`
- `noodles`

Food-101 helper command:

```bash
python ml/download_food101.py --root ml/raw_data --output-dir ml/dataset --classes pizza,hamburger,omelette --max-images-per-class 300
```

Custom dataset validation or split:

```bash
python ml/prepare_imagefolder.py --input-dir ml/raw_nigerian_foods --output-dir ml/dataset --val-split 0.15 --test-split 0.15
```

Important notes:

- Kaggle downloads may require a Kaggle account and `kaggle.json` API token.
- Check dataset license terms before academic or public use.
- Use only images you are allowed to use.
- Collect varied lighting, angles, backgrounds, and serving sizes.
- Avoid duplicates and manually review labels.
- Food-101 is useful as a baseline, but your final FoodIntel model should be retrained on Nigerian/local food classes that match the backend slugs.
