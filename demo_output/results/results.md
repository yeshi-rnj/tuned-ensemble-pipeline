# Results

> DEMO RUN on a built-in sample dataset, not the real ML-C1 data.

Best params: `{"clf__max_depth": 10, "clf__max_features": 0.5, "clf__min_samples_leaf": 1, "clf__min_samples_split": 4, "clf__n_estimators": 661, "select__threshold": 0.01}`

Features kept: **17 / 30**

## Cross-validation progression (train set)

|                                        |   cv_f1_macro_mean |   cv_std |
|:---------------------------------------|-------------------:|---------:|
| 1. Baseline (ML-C1)                    |             0.9764 |   0.0106 |
| 2. Baseline + feature selection        |             0.9413 |   0.0212 |
| 3. Random Forest + selection (untuned) |             0.9436 |   0.0173 |
| 4. Tuned ensemble                      |             0.9579 |   0.0174 |


## Final test-set comparison (evaluated once)

| model            |   test_accuracy |   test_f1_macro |
|:-----------------|----------------:|----------------:|
| Baseline (ML-C1) |          0.9825 |          0.9812 |
| Tuned ensemble   |          0.9561 |          0.9526 |


F1-macro change on test set: **-0.0285**
