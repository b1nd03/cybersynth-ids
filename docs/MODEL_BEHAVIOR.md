# Model Behavior With Multiple Datasets

## Why Different Datasets Can Work Together

Cybersecurity datasets often describe similar network-flow behavior with different column names and label styles. The preprocessing pipeline maps those sources into one shared schema before training. That schema keeps common traffic fields such as protocol, service, ports, bytes, packets, timing, flags, binary label, attack category, source dataset, environment, and subcategory.

After normalization, each row has the same structure even if it came from a different original dataset. This lets the model learn flow behavior instead of learning one file format.

## What The Model Learns

The LightGBM model predicts:

```text
0 = Normal
1 = Attack
```

Training uses traffic features only. These metadata columns are excluded from the feature matrix:

```text
dataset_source
source_file
environment_type
attack_category
attack_subcategory
label
```

Those columns are still kept for reporting and filtering. They help the website show dataset mix, category metrics, source filters, and generator controls, but they are not used as model inputs.

## Handling New Values

Numeric columns use median imputation, so missing numeric values are filled from the training data.

Categorical columns use a constant `missing` value and one-hot encoding with unknown-value handling. This means an uploaded CSV can contain a service or protocol value that did not appear in training without crashing prediction. The model can score it, but the score should be checked with drift analysis.

## Where Accuracy Comes From

Accuracy improves when:

- Raw datasets are mapped consistently into the shared schema.
- Train, validation, and test splits keep labels and attack categories represented.
- Leaky metadata is excluded from training features.
- The decision threshold is tuned on validation data.
- Metrics are checked by attack category, not only as one overall score.
- New uploaded CSV files are checked for drift before being trusted.

## When To Retrain

Retrain the model when:

- A new dataset is added.
- A source has a different feature meaning or unit scale.
- Drift monitoring shows moderate or high distribution shift.
- Per-category metrics show lower recall for an attack family.
- The generator is changed enough to affect synthetic-data experiments.

## Operational Guidance

For strongest results, validate new network sources with held-out real traffic and retrain when drift changes the feature distribution.

Generated rows are designed for controlled experiments, augmentation, balancing, and pipeline testing.

The public repository ships tiny synthetic demo splits for runtime convenience. Full processed train/test splits should be regenerated locally from approved dataset sources.
