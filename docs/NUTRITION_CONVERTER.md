# FoodIntel Nutrition Converter

FoodIntel can convert CSV or Excel nutrition tables into the backend seed format automatically.

## Script

- [backend/app/scripts/convert_food_nutrition_table.py](/Users/mac/Desktop/FoodIntel%20FullStack/backend/app/scripts/convert_food_nutrition_table.py)

## Alias Mapping

- [backend/app/data/food_slug_aliases.json](/Users/mac/Desktop/FoodIntel%20FullStack/backend/app/data/food_slug_aliases.json)

The alias file maps common food-table names to FoodIntel slugs such as:

- `fried plantain` -> `plantain`
- `moin-moin` -> `moi_moi`
- `rice, jollof` -> `jollof_rice`

## Example Usage

For Excel:

```bash
PYTHONPATH=backend .venv/bin/python -m app.scripts.convert_food_nutrition_table \
  --input /path/to/WAFCT_2019.xlsx \
  --sheet 0 \
  --output backend/app/data/food_nutrition_seed.generated.json
```

For CSV:

```bash
PYTHONPATH=backend .venv/bin/python -m app.scripts.convert_food_nutrition_table \
  --input /path/to/foods.csv \
  --output backend/app/data/food_nutrition_seed.generated.json
```

From inside `backend/`:

```bash
PYTHONPATH=. ../.venv/bin/python -m app.scripts.convert_food_nutrition_table \
  --input /path/to/foods.csv \
  --seed-input app/data/food_nutrition_seed.json \
  --aliases app/data/food_slug_aliases.json \
  --output app/data/food_nutrition_seed.generated.json
```

Then import the generated file:

```bash
PYTHONPATH=backend .venv/bin/python -m app.scripts.import_food_nutrition \
  --input backend/app/data/food_nutrition_seed.generated.json
```

From inside `backend/`:

```bash
PYTHONPATH=. ../.venv/bin/python -m app.scripts.import_food_nutrition \
  --input app/data/food_nutrition_seed.generated.json
```

## What It Does

1. Reads CSV or Excel with `pandas`
2. Detects common nutrient columns automatically
3. Maps names to FoodIntel slugs using aliases
4. Updates existing foods where possible
5. Creates new foods for unmatched rows
6. Writes a clean JSON seed file for import

## Important Note

The converter improves the nutrition database, not the image classifier itself.
The image model should still predict food classes; the backend should then map the predicted slug to the nutrition database.
