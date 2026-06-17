# Solubility Binary Classification (PyTorch MLP)

This project trains two classical multilayer perceptrons (MLP) on
`input/solub_df1_train.tsv` to predict binary `Solub_Class`.

The base dataset used was retrieved from EU-Openscreen [ECBD](https://ecbd.eu/assays/EOS300150). For each compound, two measurements were available, which were aggregated in the following way:  

Keep only one measurement from the duplicate measurements
  * in the turbidimetric assay, lower values correspond to higher solubility
  * if both values are > 0, keep the lower one
  * if both values are < 0, keep the lower one
    * does not really matter, the point is to keep a solubility `HIGH` class
  * if one is < 0 and one is > 0, keep the one > 0 (more realistic?)

The sub-folders `model_1` and `model_2` contain the best performing models from each iteration. `model_1` contains smaller molecules (up to 19 non-hydrogen atoms), `model_2` contains larger molecules (20 and more non-hydrogen atoms).  
All molecule structures were standardized with the `medchemrac` protocol of `stand_struct.py` from [jupy_tools](https://github.com/apahl/jupy_tools).

In order to maximize unbiasedness during the training process, the holdback data used for validation and the training data are separated by Murcko scaffold, i.e. any scaffold (molecular framework) that is present in the holdback validation data was completely removed from the training data.

18 Standard RDKit descriptors plus the RDKit fragment counts were used as labels, all labels were range-scaled between 0 and 1, separately for each dataset used per model (but for train and holdback combined; file names containing the term `scaled`).  
Other options for labels (different fingerprints) might be considered later, as well as a completely different network structure (e.g. [PyTorch AttentiveFP](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html)), but these require more setup.

The models were created on a VM of the Fraunhofer EdgeCloud using an Nvidia A100 graphics card. Several random seeds were explored and the model with the best balanced performance (accuracy, precision, recall and f1) was kept

## Results

### Best Model 1

* Random seed: 300
* Holdback metrics: 
    * accuracy: 0.766
    * precision: 0.933
    * recall: 0.807
    * f1: 0.865
    * roc_auc: 0.526

### Best Model 2

* Random seed: 1336
* Holdback metrics: 
    * accuracy: 0.709
    * precision: 0.942
    * recall: 0.731
    * f1: 0.823
    * roc_auc: 0.602

## Cross-Model Validation

Testing each model on the holdback data of the other model. A bit surprisingly, the models perform quite well in this test.

### Best Model 1 tested on Model 2 Holdback

* Holdback metrics: 
    * accuracy: 0.803
    * precision: 0.932
    * recall: 0.849
    * f1: 0.889
    * roc_auc: 0.542

### Best Model 2 tested on Model 1 Holdback

* Holdback metrics: 
    * accuracy: 0.798
    * precision: 0.933
    * recall: 0.844
    * f1: 0.886
    * roc_auc: 0.508

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --train-path model_1/input/solub_df1_scaled_train.tsv --holdback-path model_1/input/solub_df1_scaled_holdback.tsv --output-dir model_1/output --seed 500
```

Behavior:
- Reads train data from `model_1/input/solub_df1_scaled_train.tsv`
- Uses the requested feature columns from `features.py`
- Splits train/validation (80/20, stratified)
- Trains MLP with early stopping
- Saves model to `model_1/output/solubility_mlp.pt`
- Evaluates on `model_1/input/solub_df1_scaled_holdback.tsv`
- Saves metrics to `model_1/output/holdback_metrics.json`

Useful overrides:

```bash
python train.py --epochs 120 --batch-size 128 --lr 0.0005 --hidden-dims 512 256 128
```

## Test / Re-evaluate Saved Model

```bash
python test.py --model-path model_1/output/solubility_mlp.pt --data-path model_1/input/solub_df1_scaled_holdback.tsv
```

This writes test metrics to `model_1/output/test_metrics.json`.


## Copilot-Generated Description of the Module
E
xact flow of what train.py does, in execution order.

1. Parse CLI arguments
It requires:
- --train-path: TSV for training/validation split
- --holdback-path: TSV for final holdback evaluation
- --output-dir: where artifacts are saved

It also accepts hyperparameters (with defaults): validation fraction, seed, epochs, batch size, learning rate, weight decay, hidden layer sizes, dropout, and early-stopping patience.

2. Set reproducibility seed
It seeds NumPy and PyTorch random generators (CPU and CUDA) so splits/training are repeatable for a given seed.

3. Prepare output directory and device
- Creates output directory if missing.
- Selects CUDA if available, else CPU.

4. Load and validate training data
- Reads the training TSV via data_utils.py.
- Extracts features and label using feature list from features.py.
- Verifies required columns exist.
- Converts feature columns to numeric and errors if conversion introduces NaNs.
- Encodes binary labels to 0/1 and stores mapping metadata.
This is done through load_tsv + get_xy in data_utils.py, using FEATURE_COLUMNS and LABEL_COLUMN from features.py.

5. Split train/validation
- Performs stratified train/validation split with the configured val_size and seed.

6. Build PyTorch datasets/loaders
- Wraps standardized arrays into TensorDataset tensors (float32).
- Creates DataLoader for training (shuffled) and validation (not shuffled).
Defined in train.py.

7. Build the MLP model
- Instantiates SolubilityMLP with input_dim = number of feature columns, hidden_dims, and dropout.
- Architecture is repeating blocks:
Linear -> BatchNorm1d -> ReLU -> Dropout
then final Linear to one logit.
- Forward returns shape [batch] logits via squeeze.
Defined in model.py, instantiated in train.py.

8. Configure class-imbalance-aware loss and optimizer
- Counts positives/negatives in y_train.
- Computes pos_weight = neg_count / max(pos_count, 1.0).
- Uses BCEWithLogitsLoss(pos_weight=...).
- Uses AdamW optimizer with lr and weight_decay.
Defined in train.py.

9. Run training loop with per-epoch validation
For each epoch:
- Model in train mode.
- For each batch:
1. Move batch to device
2. Zero gradients
3. Forward pass
4. Compute BCE-with-logits loss
5. Backprop
6. Optimizer step
7. Accumulate weighted batch loss
- Compute average train loss.
- Evaluate on validation set via evaluate_model (model.eval(), no_grad()).
- evaluate_model computes:
loss, accuracy, precision, recall, f1, and roc_auc (if AUC undefined because only one class present, stores NaN).
All in train.py.

10. Track best checkpoint and early stopping
- Validation score is roc_auc; if NaN, fallback score is f1.
- If score improves:
1. Save full best_state in memory (model weights + metadata)
2. Reset bad_epochs counter
- If not improved:
1. Increment bad_epochs
2. Stop early when bad_epochs >= patience
Saved metadata includes model params, feature/label info, label mapping, scaler mean/std, and seed.
Defined in train.py.

11. Persist best model and training history
- Saves best_state to output_dir/solubility_mlp.pt
- Saves per-epoch history list to output_dir/train_history.json
Defined in train.py.

12. Evaluate best checkpoint on holdback set
- Loads holdback TSV.
- Extracts features/labels same as before.
- Scales holdback features using training mean/std (not refit).
- Reloads best model weights.
- Computes holdback metrics with evaluate_model.
- Saves to output_dir/holdback_metrics.json
- Prints rounded metrics.
Defined in train.py.

Key artifacts produced
- solubility_mlp.pt: best checkpoint + preprocessing/label metadata.
- train_history.json: epoch-by-epoch train/validation logs.
- holdback_metrics.json: final holdback performance metrics.

One subtle behavior worth noting
Label encoding in data_utils.py sorts unique label values and maps first -> 0, second -> 1. So which semantic class becomes positive depends on lexical sort order of raw label values unless your data labels are already ordered intentionally.