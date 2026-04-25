# FoodIntel Nutrition Data

Food nutrition should be handled as a structured database import, not as part of image-model training.

## Recommended Data Sources

- FAO/INFOODS Food Composition Table for Western Africa (2019):
  - https://www.fao.org/food-composition/tables-and-databases-2/detail/food-composition-tables/en
- FAO/INFOODS database directory:
  - https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/
- USDA FoodData Central:
  - https://fdc.nal.usda.gov/

## Practical Rule

- Train the model on images to predict a `slug` such as `plantain` or `egusi_soup`.
- Store nutrition values separately in the `foods` collection.
- After prediction, map the predicted `slug` to a food record and return its nutrition values.

This is the correct end-to-end architecture because nutrient values are database data, not image labels.

## Current FoodIntel Seed

FoodIntel now keeps its nutrition seed in:

- [backend/app/data/food_nutrition_seed.json](/Users/mac/Desktop/FoodIntel%20FullStack/backend/app/data/food_nutrition_seed.json)

You can import or refresh it with:

```bash
PYTHONPATH=backend .venv/bin/python -m app.scripts.import_food_nutrition \
  --input backend/app/data/food_nutrition_seed.json
```

Or from inside `backend/`:

```bash
PYTHONPATH=. ../.venv/bin/python -m app.scripts.import_food_nutrition \
  --input app/data/food_nutrition_seed.json
```

## Extending Nutrition Data

1. Download a structured nutrition table such as the FAO Western Africa table in Excel/CSV.
2. Normalize food names to FoodIntel slugs like `jollof_rice`, `plantain`, `okro_soup`.
3. Convert the source data into the JSON shape used by `food_nutrition_seed.json`.
4. Re-run the import script.

Both `/foods/seed` and `import_food_nutrition.py` now validate the same seed structure before writing to MongoDB.

## Important Note

Do not try to "train nutrition into the image model". The model should classify the food image; the backend should supply nutrition from the food database.
