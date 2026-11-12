# Marine FNO Training and Evaluation

## 1. Overview

The marine Fourier Neural Operator (FNO) predicts six-hour water-level and current fields over the regional Mombasa marine grid.

The current frozen experimental baseline is:

- **Model version:** `fno-0.4.0`
- **Prediction mode:** `residual`
- **Contract version:** `2.0`
- **Architecture:** `MarineFNO`
- **Input channels:** 6
- **Output fields:** 3
- **Architecture configuration:** width 24, modes_x 8, modes_y 8, 4 layers
- **Forecast horizon:** 6 hours
- **Forecast interval:** 1 hour
- **Device:** CPU
- **Checkpoint:** `artifacts/marine_fno_residual.pt`

The model predicts changes relative to the initial water-level and current fields. These predicted residuals are added to the initial fields to produce absolute physical forecasts.

**Model status:** Frozen experimental baseline for MSA Port Twin integration. This does not indicate validated operational forecasting accuracy.

## 2. Dataset

Training uses the prepared dataset:

`data/processed/marine/mombasa_training_dataset.npz`

The accompanying JSON manifest records dataset provenance and preparation metadata.

| Property                   | Value                        |
| -------------------------- | ---------------------------- |
| Geographic area            | Regional Mombasa marine area |
| Coordinate system          | EPSG:4326                    |
| Grid                       | 23 × 10                      |
| Approximate resolution     | 9 km                         |
| Time coverage              | 1–30 October 2026            |
| Temporal resolution        | 1 hour                       |
| Forecast horizon           | 6 hours                      |
| Generated samples          | 714                          |
| Purged overlapping samples | 12                           |
| Retained samples           | 702                          |
| Training samples           | 493                          |
| Validation samples         | 101                          |
| Test samples               | 108                          |

### Data provenance

- Copernicus Marine: `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`, version 202406 (MOI GLO12).
- GEBCO_2026: bathymetry.

The Copernicus Marine fields are gridded model products, not independent in-situ measurements. The dataset therefore supports model development and evaluation against those products, not independent verification of ocean forecasting accuracy.

The regional grid is approximately 9 km resolution and does not resolve detailed harbour-channel conditions.

The source vertical references also differ: marine water level uses a geoid reference, while GEBCO elevation uses height above mean sea level (EPSG:5831). No datum transformation was applied.

The chronological dataset split includes purging of samples whose forecast windows cross partition boundaries. However, the resulting samples remain temporally correlated and should not be treated as statistically independent observations.

## 3. Model input and prediction contract

The model accepts six input channels:

1. Bathymetry (positive downward, metres).
2. Initial water level.
3. Initial eastward current.
4. Initial northward current.
5. Normalized longitude.
6. Normalized latitude.

It predicts three physical fields over six forecast steps:

- `water_level_m`
- `current_u_ms`
- `current_v_ms`

The output shape is:

`[batch, 6, 3, ny, nx]`

Input and target normalization statistics are fitted using the training partition. The checkpoint stores the relevant normalization statistics and prediction mode.

In residual mode, the model predicts changes relative to the initial physical fields. The initial fields are added back after residual prediction and denormalization, so inference returns absolute physical values.

## 4. Training configuration and experiments

Training uses the existing `train_model` implementation with Adam optimization and normalization fitted on the training partition.

### Historical experiments

Earlier experiments explored absolute prediction and different learning rates.

| Experiment                  | Model version | Learning rate | Best epoch | Best validation loss |
| --------------------------- | ------------- | ------------: | ---------: | -------------------: |
| Seeded experiment           | `fno-0.2.0`   |         0.001 |          5 |             0.250437 |
| Selected experiment         | `fno-0.2.0`   |        0.0001 |         19 |             0.217247 |
| Earlier unseeded experiment | `fno-0.2.0`   |         0.001 |          4 |             0.238952 |

These results are retained as historical experiments and are not the selected frozen baseline.

### Frozen residual experiment

The selected residual experiment used:

| Setting              | Value       |
| -------------------- | ----------- |
| Model version        | `fno-0.4.0` |
| Prediction mode      | `residual`  |
| Epochs               | 20          |
| Learning rate        | 0.001       |
| Optimizer            | Adam        |
| Device               | CPU         |
| PyTorch seed         | 42          |
| Best epoch           | 3           |
| Best validation loss | 0.590043    |

The checkpoint was saved to:

`artifacts/marine_fno_residual.pt`

The checkpoint preserves the model parameters, architecture and relevant input/target normalization statistics, prediction mode, and model metadata.

The residual experiment substantially improved on the earlier absolute-prediction model. However, its validation loss should not be compared directly with the historical absolute-model losses as if they represented the same prediction target: the residual model is trained against a different target representation.

## 5. Test evaluation

The frozen residual checkpoint was evaluated against the 108 samples in the test partition.

Metrics are calculated on absolute physical forecast values. The baseline is persistence: it repeats the initial water-level and current fields across all six forecast steps.

| Variable        |  FNO MAE | FNO RMSE | Persistence MAE | Persistence RMSE |
| --------------- | -------: | -------: | --------------: | ---------------: |
| Water level (m) | 0.003699 | 0.004986 |        0.003413 |         0.004669 |
| Current U (m/s) | 0.027726 | 0.036941 |        0.026880 |         0.036250 |
| Current V (m/s) | 0.030912 | 0.042533 |        0.029734 |         0.041893 |

The residual model performs substantially better than the earlier absolute-prediction model. However, it remains slightly worse than persistence on all three variables in these aggregate test metrics.

The test results therefore do not establish that the FNO outperforms the persistence baseline overall.

Validation results showed improvements over persistence for current components across the six forecast hours, with less consistent improvements for water level. These validation results are diagnostic and do not override the aggregate test results.

## 6. Inference verification

The residual checkpoint is compatible with the existing marine inference pipeline.

Inference reads the prediction mode and normalization statistics from the checkpoint, performs residual prediction, and returns absolute physical forecast values.

The expected output for each field is:

- Water level: `(6, 23, 10)`
- Current U: `(6, 23, 10)`
- Current V: `(6, 23, 10)`

All output values must be finite.

Inference verification establishes checkpoint compatibility and successful execution. It does not establish real-world forecasting accuracy.

## 7. Limitations and intended use

The following limitations apply to the frozen experimental baseline:

- The training period covers only 30 days.
- Copernicus Marine products are model-derived fields rather than independent in-situ ground truth.
- The approximately 9 km grid is too coarse for detailed harbour-channel hydrodynamics.
- The marine and bathymetry vertical references differ, and no datum transformation was applied.
- Temporal correlation between samples limits how independently evaluation results can be interpreted.
- Aggregate test performance remains slightly worse than persistence for all three output variables.
- Validation improvements have not consistently translated into test improvements.
- No independent Mombasa field-observation validation has been performed.

### Intended use

The model can be integrated into MSA Port Twin as an experimental regional marine prediction capability.

Its outputs may support:

- Marine prediction visualizations.
- Experimental water-level and current-field forecasts.
- Digital-twin demonstrations and model integration.
- Further evaluation and training improvements.

The model must not be treated as a validated source for vessel navigation, safe-draft decisions, or actual port operational scheduling.

Predictions presented in the application should clearly identify:

- Model version: `fno-0.4.0`.
- Prediction mode: `residual`.
- Forecast horizon: 6 hours.
- Model status: experimental.
- Data source and input provenance.
- Prediction timestamp.

## 8. Frozen baseline decision

`fno-0.4.0` is the frozen experimental baseline selected for integration into MSA Port Twin.

The model implementation, checkpoint, inference compatibility and evaluation are retained as the current reference point for future improvements.

Further work on data quality, vertical-datum reconciliation, training strategies, evaluation and independent validation can continue separately without blocking application integration.

Any future model improvements should be evaluated against this frozen baseline and the persistence benchmark before replacing it.
