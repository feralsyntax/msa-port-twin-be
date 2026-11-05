cat > docs/marine-fno-training.md <<'EOF'

# Marine FNO Training and Evaluation

## 1. Overview

The marine Fourier Neural Operator (FNO) is trained to predict six-hour
water-level and current fields over the regional Mombasa marine grid.

- Model version: `fno-0.2.0`
- Contract version: `2.0`
- Architecture: `MarineFNO`
- Input channels: 6
- Output channels: 3
- Architecture configuration: width 24, modes_x 8, modes_y 8, 4 layers
- Forecast horizon: 6 hours
- Device: CPU

## 2. Dataset

Training uses the prepared dataset:

`data/processed/marine/mombasa_training_dataset.npz`

The accompanying JSON manifest records dataset provenance and preparation
metadata.

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

- Copernicus Marine: `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`,
  version 202406 (MOI GLO12).
- GEBCO_2026: bathymetry.

The Copernicus Marine fields are gridded model products, not independent
in-situ measurements. The dataset therefore supports model development and
evaluation against those products, not independent verification of ocean
forecasting accuracy.

The regional grid is approximately 9 km resolution and does not resolve
detailed harbour-channel conditions. The source vertical references also
differ: marine water level uses a geoid reference, while GEBCO elevation
uses height above mean sea level (EPSG:5831). No datum transformation was
applied.

## 3. Training configuration and experiments

Training uses the existing `train_model` implementation with Adam
optimization and target normalization fitted on the training partition.

All controlled experiments below use 20 epochs, CPU execution, and
PyTorch seed 42.

| Experiment          | Learning rate | Best epoch | Best validation loss |
| ------------------- | ------------: | ---------: | -------------------: |
| Seeded experiment   |         0.001 |          5 |             0.250437 |
| Selected experiment |        0.0001 |         19 |             0.217247 |

The lower learning rate produced approximately 13.3% lower validation loss
than the seeded 0.001 experiment and showed more stable training.

An earlier unseeded 0.001 experiment achieved validation loss 0.238952 at
epoch 4, but it is not used for the controlled comparison.

### Selected checkpoint

`artifacts/marine_fno_lr1e-4.pt`

The checkpoint stores the model architecture, model and contract versions,
model parameters, and target normalization statistics.

## 4. Test evaluation

The selected checkpoint was evaluated once against the untouched test
partition containing 108 samples.

Metrics are reported as mean absolute error (MAE) and root mean squared
error (RMSE).

| Variable        |  FNO MAE | FNO RMSE | Baseline MAE | Baseline RMSE |
| --------------- | -------: | -------: | -----------: | ------------: |
| Water level (m) | 0.014165 | 0.016803 |     0.003413 |      0.004669 |
| Current U (m/s) | 0.105729 | 0.134344 |     0.155569 |      0.201413 |
| Current V (m/s) | 0.258035 | 0.284524 |     0.299689 |      0.363242 |

The baseline uses water-level persistence and zero current components.

The FNO improves both current-component metrics compared with the baseline,
but performs worse for water level. It therefore does not demonstrate
overall superiority over the baseline.

## 5. Inference verification

The selected checkpoint was loaded through the existing
`MarineInferenceService` and used to predict from a test sample.

Verification results:

- Status: `success`
- Model version: `fno-0.2.0`
- Contract version: `2.0`
- Water-level output: `(6, 23, 10)`
- Current-U output: `(6, 23, 10)`
- Current-V output: `(6, 23, 10)`
- All output values were finite.

This verifies checkpoint compatibility and inference execution, not
real-world forecasting accuracy.

## 6. Limitations

- The training period covers only 30 days.
- Copernicus Marine products are model-derived fields rather than
  independent in-situ ground truth.
- The approximately 9 km grid is too coarse for detailed harbour-channel
  hydrodynamics.
- The marine and bathymetry vertical references differ, and no datum
  transformation was applied.
- Temporal correlation between samples limits how independently the
  evaluation samples can be interpreted.
- The baseline is deliberately simple and uses zero current components.
- Water-level prediction currently underperforms the baseline.
- No independent field-observation validation has been performed.

The trained model should therefore be treated as an experimental regional
prediction model, not a validated operational marine forecasting system.
EOF
