# BE52 — Marine Data Audit: Tide, Domain, Resolution, Provenance and Vertical Datum

**Project:** MSA Port Twin — Port of Mombasa  
**Audit scope:** Existing marine datasets and their suitability for modelling the Mombasa Port and Kilindini Channel  
**Status:** Investigation complete; written audit consolidation  
**Code changes:** None  
**Data changes:** None

---

## 1. Executive summary

This audit assessed the existing Copernicus Marine, GEBCO and Mombasa tide-gauge data against the requirements of a port-focused marine digital twin.

The findings identify three significant limitations:

1. **Tides:** The existing Copernicus GLO12 sea-surface-height field does not reproduce Mombasa's astronomical tidal signal.
2. **Spatial suitability:** The Copernicus grid is too coarse for harbour-scale modelling. GEBCO provides finer coverage, but its resolution and source provenance do not establish suitability for detailed channel or berth depths.
3. **Vertical datum:** The relationship between the existing Copernicus and GEBCO vertical references and Mombasa Chart Datum has not been established. The prepared dataset also contains inconsistent vertical CRS metadata.

The Mombasa tide-gauge record provides a useful independent reference. Its fitted M2 and S2 tidal constituents closely match the published KMFRI values.

**Overall decision:** The existing products are useful for regional marine context and preliminary experimentation, but they are not sufficient as a validated, internally consistent source of tide and bathymetry for operational Kilindini Channel modelling.

## 2. Audit scope and evidence

The audit covered:

- Copernicus Marine GLO12 hourly surface fields.
- GEBCO 2026 bathymetry and its Type Identifier (TID) grid.
- Prepared Mombasa and offshore training dataset metadata.
- Historical Mombasa tide-gauge observations from UHSLC.
- Published Mombasa tidal constituents and tide-gauge benchmark relationships.
- Spatial coverage and resolution within the published Admiralty Chart 666 bounds used for the port-window assessment.

The inspected Copernicus dataset was:

`cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`

The inspected GEBCO products were:

- `gebco_2026_mombasa_extended.nc`
- `gebco_2026_mombasa_extended_tid.nc`

The prepared datasets were:

- `mombasa_training_dataset.npz` and its JSON metadata.
- `offshore_training_dataset.npz` and its JSON metadata.

## 3. Tide audit

### 3.1 Copernicus GLO12 assessment

The Copernicus `zos` variable represents sea-surface height above the geoid. The inspected product is an hourly GLO12 model product.

A harmonic analysis was performed using 720 hourly observations covering October 2026 at the grid point nearest the Mombasa tide gauge:

- Gauge reference: approximately 39.657247°E, 4.069663°S.
- Selected Copernicus grid point: approximately 39.666672°E, 4.083328°S.

Results:

| Metric                      |     Result |
| --------------------------- | ---------: |
| Samples                     |        720 |
| Mean sea-surface height     |  0.48353 m |
| Standard deviation          |  0.01617 m |
| Observed range              |  0.08723 m |
| Residual standard deviation | 0.014485 m |
| Harmonic fit R²             |    0.19749 |

Fitted constituent amplitudes:

| Constituent | Copernicus amplitude |
| ----------- | -------------------: |
| M2          |           0.000493 m |
| S2          |           0.002719 m |
| N2          |           0.000754 m |
| K1          |           0.003778 m |
| O1          |           0.002004 m |
| K2          |           0.000245 m |

The combined M2 and S2 spring-range estimate is approximately 0.00642 m.

These amplitudes are negligible compared with the independently established Mombasa tidal constituents.

The Copernicus quality documentation explicitly distinguishes GLO12 dynamic sea-surface-height variations from the tidal contribution. A separate Copernicus merged sea-level product includes ocean and load tide contributions.

**Decision: NO-GO for astronomical tidal prediction using the existing GLO12 `zos` field.**

GLO12 may remain useful for regional ocean-state context, but its `zos` field must not be treated as a substitute for a Mombasa tidal model.

### 3.2 Independent Mombasa tide-gauge assessment

The UHSLC research-quality hourly Mombasa record was inspected.

Station details:

- Station: Mombasa.
- UHSLC ID: 101.
- GLOSS ID: 8.
- Location: approximately 39.65732°E, 4.06999°S.
- Coverage: 17 June 1986 to 31 December 2024.
- Valid observations: 263,399.
- Missing observations: 74,463.
- Overall missing-data proportion: approximately 22%.
- Reference level: station zero.

The 2014–2019 period was selected for harmonic analysis because it provides a multi-year interval with comparatively good coverage.

The simplified least-squares harmonic fit produced:

| Constituent | Fitted gauge amplitude | KMFRI published amplitude |
| ----------- | ---------------------: | ------------------------: |
| M2          |              1.07405 m |                  1.0457 m |
| S2          |              0.51445 m |                  0.5129 m |
| N2          |              0.19450 m |                  0.1923 m |
| K1          |              0.17407 m |                  0.1928 m |
| O1          |              0.09584 m |                  0.1145 m |
| K2          |              0.11212 m |                  0.1393 m |

Additional results:

- Valid observations used: 52,418.
- Residual standard deviation: 0.09517 m.
- R²: 0.98814.
- Gauge M2 + S2 spring-range estimate: approximately 3.17698 m.
- KMFRI M2 + S2 spring-range estimate: approximately 3.1172 m.

The fitted M2 amplitude differs from the published value by approximately 2.7%; S2 differs by approximately 0.3%.

This provides a useful independent check that the gauge record reproduces the published Mombasa tidal characteristics.

**Decision: GO for preliminary tidal characterization and independent reference validation.**

The analysis is not a professional tidal prediction solution. It uses a simplified fixed-frequency least-squares fit without nodal corrections, astronomical argument corrections, constituent inference or formal uncertainty estimates. The gauge and Copernicus periods are also not simultaneous, so this is a comparison of tidal characteristics rather than point-by-point validation.

### 3.3 Alternative Copernicus product

The Copernicus merged sea-level dataset:

`cmems_mod_glo_phy_anfc_merged-sl_PT1H-i`

is a candidate for further evaluation because its documented components include ocean tide and load tide contributions.

It has not been downloaded, tested or validated against the Mombasa gauge in this audit.

**Decision: CONDITIONAL GO for evaluation as a candidate tidal input. No claim of Mombasa suitability is made.**
