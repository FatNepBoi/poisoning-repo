# Reproducing the Experiments

> **Note:** The repository already contains all required models and datasets. The steps below are only needed if you want to reproduce everything from scratch.

## Before You Start

To reproduce the experiments from the beginning, **keep only the contents of `\Datasets\Normal Datasets\`** and delete everything else inside the other folders.

> **Do not delete any folders, only their contents.**

## Steps

1. **Train the initial models**
   - Make sure `\extraction_models` is empty.
   - Run `Train_Initial_Models.ipynb` with `SAVE_NORMAL_MODELS = True` (third cell of the notebook).
   - If everything went correctly, `\extraction_models` now contains **30 `.pth` files**.

2. **Create the poisoned datasets**
   - Run `Create_Poisoning.ipynb`.

3. **Train and evaluate the poisoned models**
   - Run `Poisoned_Models_Tau_{25, 50, 100}.ipynb` (one notebook per tau value).
   - This trains all the models and produces the resulting graphs.

## Configuration Flags in `Poisoned_Models_Tau_{25, 50, 100}.ipynb`

The recommended setup for a first run, for reproducibility purposes:

```python
SAVE_NORMAL_MODELS = True
SAVE_CALCULATED_MODELS = True
SAVE_GAUSSIAN_MODELS = True
SAVE_UNIFORM_MODELS = True

TRAIN_NORMAL_MODELS = True
TRAIN_CALCULATED_MODELS = True
TRAIN_GAUSSIAN_MODELS = True
TRAIN_UNIFORM_MODELS = True

LOAD_NORMAL_MODELS = False
LOAD_CALCULATED_MODELS = False
LOAD_GAUSSIAN_MODELS = False
LOAD_UNIFORM_MODELS = False

SHOW_PREDICTIONS_GRAPHS = True
```

| Flag prefix | Effect |
|-------------|--------|
| `SAVE_*` | Saves the corresponding models to `.pth` files after training. |
| `TRAIN_*` | Trains the corresponding models. |
| `LOAD_*` | Loads the corresponding models from `.pth` files before testing. |
| `SHOW_PREDICTIONS_GRAPHS` | Prints the testing graphs in the last cell of the notebook. |

These flags exist so you don't have to retrain everything if a change is made halfway through.

> **Rule:** if a `LOAD_*` flag is `True`, the matching `SAVE_*` and `TRAIN_*` flags will logically be `False`.

## Note on Seeds

The seeds used are found within the code. There are two sets of seeds:

- one for the **gradient extraction and noise design**
- one for the **poisoned runs**