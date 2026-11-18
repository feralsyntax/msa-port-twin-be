# BE57 — Independent Marine Model Validation Assessment

**Project:** MSA Port Twin — Port of Mombasa  
**Scope:** Independent validation of Marine FNO V2 against Mombasa observations  
**Model:** Marine FNO V2 (`fno-0.5.0`)  
**Dataset:** V2 synthetic shallow-water scenarios  
**Status:** Independent real-world validation blocked — evidence assessment complete  
**Code changes:** None  
**Model changes:** None  
**Production integration:** Out of scope

---

## 1. Executive summary

BE57 investigated whether the Marine FNO V2 model could be independently validated against historical observations from Mombasa.

An independent UHSLC research-quality tide-gauge dataset was obtained and inspected. Its timestamps, quality flags, missing values, measurement range, reference datum and geographical location were checked. The gauge location was also mapped to the V2 model grid.

However, the available V2 scenarios are synthetic simulations rather than historical reconstructions of observed marine conditions. They use prescribed synthetic forcing, zero initial water-level anomaly and zero initial currents. The available historical gauge record does not provide the spatial initial conditions or matching historical forcing required to reproduce those scenarios.

Consequently, comparing the V2 predictions directly with the historical gauge measurements would not constitute a fair forecast-validation experiment.

**Overall decision: Independent real-world validation is currently blocked.**

The gauge dataset is a useful independent observation source, but the available evidence does not establish real-world forecast skill for water level or currents. No calibrated real-world uncertainty estimate can be reported.

## 2. Model and validation scope

Marine FNO V2 has the following characteristics:

- Model version: `fno-0.5.0`
- Dataset version: `v2`
- Input contract: `marine-fno-v2-1.0`
- Input channels: 13
- Output variables:
  - Water level (`water_level_m`)
  - Eastward current (`current_u_ms`)
  - Northward current (`current_v_ms`)
- Forecast horizon: six hourly steps
- Output shape: `[batch, 6, 3, ny, nx]`

The intended independent validation objectives were:

1. Compare forecast water levels against independent tide-gauge observations.
2. Compare predicted currents against independent current measurements.
3. Report forecast skill separately for each variable and lead time.
4. Establish a simple uncertainty indicator that communicates the reliability of the results.

These objectives require observations and model inputs representing the same historical conditions.

## 3. Independent tide-gauge observations

### 3.1 Data source

The independent water-level observations were obtained from the University of Hawaiʻi Sea Level Center (UHSLC) Research Quality Data service.

| Property             | Value                    |
| -------------------- | ------------------------ |
| Station              | Mombasa                  |
| UHSLC ID             | 101                      |
| GLOSS ID             | 8                        |
| Observation period   | 2020-01-01 to 2024-12-31 |
| Sampling interval    | Hourly                   |
| Measurement variable | Sea level                |
| Measurement units    | Millimetres              |
| Reference datum      | Station zero             |
| Station longitude    | 39.65732°E               |
| Station latitude     | 4.06999°S                |
| Quality flag         | 4                        |

### 3.2 Dataset integrity and quality

Raw file:

`data/raw/observations/mombasa_uhslc_hourly_2020_2024.csv`

| Check                           |                   Result |
| ------------------------------- | -----------------------: |
| Data rows                       |                   43,848 |
| Expected hourly timestamps      |                   43,848 |
| Duplicate timestamps            |                        0 |
| Timestamp gaps                  |                        0 |
| Finite sea-level observations   |                   39,021 |
| Missing/non-finite observations |                    4,827 |
| Finite observation proportion   |      Approximately 89.0% |
| Quality flag                    | 4 for all inspected rows |

Finite observation statistics:

| Statistic          |     Result |
| ------------------ | ---------: |
| Minimum            |   1,000 mm |
| Maximum            |   5,287 mm |
| Mean               | 2,994.3 mm |
| Standard deviation |   850.2 mm |

The quality flag does not make missing measurements valid. Non-finite values must be excluded from any subsequent statistical comparison.

The original downloaded CSV is retained unchanged. Its SHA-256 checksum is:

`ff9f8c85692802c66285688e5ec1e2d10c19fb4d56bbdeac922d51b3f51040c0`

The observations are suitable as an independent historical water-level reference, subject to datum interpretation and appropriate handling of missing values.

### 3.3 Vertical datum considerations

The UHSLC record identifies its reference datum as station zero.

The BE52 audit records the BODC benchmark relationship:

`TGZ = Admiralty Chart Datum - 0.942 m`

It also records the corresponding proposed relationship:

`H_ACD = H_TGZ - 0.942 m`

This relationship must not be applied automatically. Its applicability and sign convention must be confirmed against the relevant station benchmark documentation and the specific observation record before any conversion.

The V2 scenarios describe water level relative to an assumed mean-sea-level reference. The available evidence does not establish a fully verified transformation between that model reference and the gauge station-zero datum.

Therefore, no datum correction was applied for this assessment.

## 4. Mapping the gauge to the V2 grid

The gauge coordinates were transformed from EPSG:4326 to the V2 grid CRS, EPSG:32737.

| Property                |            Result |
| ----------------------- | ----------------: |
| Projected easting       |      572,961.62 m |
| Projected northing      |    9,550,106.52 m |
| Nearest grid cell       | `(iy=115, ix=46)` |
| Grid-cell easting       |         573,000 m |
| Grid-cell northing      |       9,550,000 m |
| Distance to cell centre |          113.22 m |
| Cell classification     |             Water |
| Cell bathymetry         |           7.754 m |

The gauge is geographically close to a water cell in the inspected V2 grid.

This confirms that spatial mapping is feasible. It does not establish that the model represents the actual hydrodynamic conditions at the gauge.

The bathymetry value is GEBCO-derived and has not been independently verified at this location. The vertical datum relationship also remains unresolved.

## 5. Why independent forecast validation is blocked

### 5.1 V2 scenarios are synthetic

The V2 training and evaluation scenarios are generated using a two-dimensional nonlinear shallow-water solver.

Their documented properties include:

- `data_classification=simulated`
- `is_observed=0`
- `is_ground_truth=0`
- Initial water-level anomaly set to zero.
- Initial currents set to zero.
- Prescribed synthetic wind forcing.
- Prescribed synthetic tidal boundary forcing.
- GEBCO 2026-derived bathymetry.
- EPSG:32737 projected grid.
- 250 m grid spacing.
- Twelve-hour simulated duration.

The scenarios do not represent historical reconstructions of the conditions observed at Mombasa during 2020–2024.

### 5.2 Historical observations do not provide matching model inputs

A tide gauge provides a time series of water level at one location. It does not provide:

- Spatially distributed initial water levels.
- Spatially distributed initial eastward and northward currents.
- Historical wind fields matching the observation period.
- Verified historical tidal boundary conditions for the model domain.
- Independently verified channel-scale bathymetry and vertical datum alignment.

The available Copernicus marine files inspected in the earlier BE52 audit cover October 2026, rather than the 2020–2024 gauge observation period. They cannot supply matching historical forcing for the downloaded gauge record.

The inspected Copernicus GLO12 `zos` field was also found unsuitable as a substitute for Mombasa astronomical tidal forcing.

### 5.3 Direct comparison would be misleading

Pairing a synthetic V2 forecast with an unrelated historical gauge measurement would mix different initial conditions, forcing, dates and potentially vertical references.

Any resulting error metric would not measure the model's ability to forecast real Mombasa conditions.

For this reason, no real-world MAE, RMSE, correlation, bias or skill score is reported.

## 6. Current prediction validation

No independent ADCP or equivalent current-observation dataset has been established for this assessment.

The tide-gauge record cannot validate eastward or northward current predictions.

Consequently:

- Eastward-current forecast skill: not established.
- Northward-current forecast skill: not established.
- Current forecast uncertainty: not calibrated.

The absence of an identified current dataset in this assessment should not be interpreted as proof that no suitable observations exist.

## 7. Synthetic evaluation versus independent validation

The existing V2 evaluator, `evaluation_v2.py`, compares FNO predictions with persistence and a harmonic-tide baseline using held-out synthetic scenarios.

Its documented scope explicitly states that:

- Evaluation uses simulated V2 held-out scenarios.
- The harmonic-tide baseline is spatially uniform.
- The harmonic baseline keeps currents persistent.
- Results do not establish real-world accuracy.

These results remain useful for assessing performance within the synthetic scenario distribution.

They must not be presented as independent validation against observed Mombasa conditions.

The existing evaluator aggregates metrics by variable over water cells and forecast steps. It does not currently provide independent real-world skill by lead time.

No changes to its evaluation logic were made as part of this assessment.

## 8. Uncertainty and validation status

A calibrated statistical uncertainty estimate cannot be derived from the available evidence because independent forecast errors have not been established.

For the current model release, use the following qualitative validation indicator:

**REAL-WORLD VALIDATION: NOT ESTABLISHED**

| Validation target                       | Status    | Reason                                           |
| --------------------------------------- | --------- | ------------------------------------------------ |
| Independent water-level observations    | Available | UHSLC Mombasa gauge record obtained              |
| Gauge data quality inspection           | Complete  | Coverage, flags and missing values checked       |
| Gauge-to-grid mapping                   | Complete  | Nearest water cell identified                    |
| Verified gauge-to-model datum alignment | Blocked   | Vertical reference transformation unresolved     |
| Historical model initial conditions     | Blocked   | Spatial initial state unavailable                |
| Matching historical forcing             | Blocked   | No matching historical forcing assembled         |
| Water-level forecast skill              | Blocked   | No valid historical hindcast comparison          |
| Eastward-current forecast skill         | Blocked   | Independent current observations not established |
| Northward-current forecast skill        | Blocked   | Independent current observations not established |
| Real-world uncertainty calibration      | Blocked   | No independent forecast-error distribution       |

The qualitative indicator communicates the evidence status only. It is not a probability, confidence interval or calibrated estimate of forecast error.

Synthetic evaluation results may still be reported separately, with an explicit synthetic-only qualification.

## 9. Requirements to unblock independent validation

A future validation experiment requires the following evidence and preparation:

1. **Historical hydrodynamic inputs:** Spatial initial water-level and current fields representative of the selected observation period.
2. **Historical forcing:** Matching wind fields and suitable tidal boundary conditions.
3. **Vertical datum alignment:** A verified relationship between gauge station zero, the model water-level reference and the bathymetric vertical reference.
4. **Bathymetric suitability:** Evidence that the bathymetry and grid are appropriate for the location being evaluated.
5. **Independent water-level observations:** Quality-controlled gauge measurements with timestamps aligned to forecast valid times.
6. **Independent current observations:** Suitable ADCP or equivalent measurements for evaluating eastward and northward currents.
7. **Lead-time evaluation:** MAE, RMSE and bias calculated separately for each output variable and forecast lead time, using consistent valid observations.
8. **Uncertainty assessment:** Forecast-error distributions and, where appropriate, calibrated prediction intervals evaluated on independent data.

The validation data must be kept separate from model training and model selection to avoid data leakage.

## 10. Safety and operational limitations

The model has not been independently validated against historical Mombasa hydrodynamic observations.

Its outputs must therefore be treated as experimental model predictions.

They must not be used as the sole basis for:

- Vessel navigation or routing.
- Safe-draft or under-keel-clearance decisions.
- Berthing or channel-entry decisions.
- Operational tidal-window decisions.
- Safety-critical marine operations.

Independent hydrographic surveys, authoritative tide information and appropriate operational measurements remain necessary for such decisions.

## 11. Final BE57 outcome

The investigation achieved the independent observation acquisition, data-quality inspection and gauge-to-grid mapping portions of BE57.

It also established why the available observations cannot currently support a scientifically defensible real-world V2 forecast-skill assessment.

**Final outcome: Validation readiness assessed; independent real-world forecast validation blocked by missing matching historical hydrodynamic inputs, unresolved datum alignment and unavailable independent current observations.**

No model accuracy claim or calibrated real-world uncertainty claim is made.

No model architecture, training, inference or production API changes were made.

---

## 12. References

1. UHSLC — [Research Quality Data](https://uhslc.soest.hawaii.edu/erddap/).
2. BODC / GLOSS — [Mombasa Station Information Sheet](https://www.bodc.ac.uk/resources/inventories/gloss_handbook/stations/8/).
3. BODC — [Mombasa Site History](https://www.bodc.ac.uk/data/documents/nodb/57529/).
4. BE52 — `docs/research/BE52-marine-data-audit.md`, including the Mombasa tide-gauge, bathymetry, tidal forcing and vertical datum assessment.
5. BE57 — `port_twin_app/services/marine_ai/evaluation_v2.py`, documenting the scope and limitations of synthetic V2 evaluation.
