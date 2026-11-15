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

## 4. Domain and spatial-resolution audit

### 4.1 Port reference window

The spatial assessment used the published bounds associated with Admiralty Chart 666:

| Boundary | Coordinate |
| -------- | ---------: |
| West     |  39.5947°E |
| East     |  39.7187°E |
| South    |   4.1052°S |
| North    |   4.0317°S |

These bounds define the chart window used for the audit. They do not describe the exact geometry of the navigable channel.

### 4.2 Copernicus GLO12

The inspected Copernicus grid has:

- Full longitude extent: 39°E–41°E.
- Full latitude extent: 5°S–3°S.
- Grid dimensions: 25 × 25.
- Approximate spacing: 9.3 km.

Within the port chart window, only one grid intersection was available.

This is inadequate for representing channel geometry, harbour circulation, berth approaches or other harbour-scale hydrodynamic features.

**Decision: NO-GO for Kilindini Channel-scale spatial modelling.**

### 4.3 GEBCO 2026

The inspected GEBCO subset has:

- Full longitude extent: approximately 38.99375°E–41.00625°E.
- Full latitude extent: approximately 5.00625°S–2.99375°S.
- Nominal grid spacing: 15 arc-seconds.
- Approximate local spacing: 463 m.

Within the port chart window:

- Grid dimensions: 17 × 29.
- Total grid cells: 493.

Elevation statistics:

| Statistic          |   Result |
| ------------------ | -------: |
| Minimum            |   −151 m |
| Maximum            |    +70 m |
| Mean               | +10.63 m |
| Median             |    +16 m |
| Standard deviation |  28.51 m |

The positive and negative values show that the chart window includes both land and seabed. Consequently, the overall mean and median are not meaningful summaries of harbour depth.

The grid is substantially finer than Copernicus GLO12, but approximately 463 m spacing is still too coarse to establish detailed channel or berth depths.

### 4.4 GEBCO Type Identifier provenance

The TID grid provides source classifications for the corresponding GEBCO elevation cells.

Observed distribution:

| TID | Classification       | Cells | Share |
| --: | -------------------- | ----: | ----: |
|   0 | Land                 |   319 | 64.7% |
|  10 | Singlebeam           |     2 |  0.4% |
|  16 | Optical              |    50 | 10.1% |
|  40 | Predicted altimetry  |    87 | 17.6% |
|  44 | Bathymetric sounding |    35 |  7.1% |

Of the 174 non-land cells:

- 2 are classified as singlebeam.
- 50 are classified as optical.
- 87 are classified as predicted altimetry.
- 35 are classified as bathymetric sounding.

There are no cells classified as multibeam (TID 11) in the inspected window.

These source classifications do not establish measurement accuracy or uncertainty. In particular, a sounding-related classification should not be interpreted as proof that the entire grid cell represents a recent, high-resolution hydrographic survey.

GEBCO itself warns that the grid should not be used for navigation or safety-at-sea purposes and that its nominal resolution can differ substantially from the resolution of the underlying measurements.

**Decision: CONDITIONAL GO for regional bathymetric context and exploratory visualisation. NO-GO for validated channel depths, safe-draft decisions or navigation-related use.**

### 4.5 Prepared Mombasa training grid

The prepared Mombasa training grid has bounds:

- West: 40.16667°E.
- East: 40.91667°E.
- South: 4.91666°S.
- North: 3.08333°S.
- Dimensions: 10 × 23.

This grid lies east of the port chart window and does not cover Mombasa Port or Kilindini Channel.

The offshore training dataset also explicitly describes its domain as Mombasa-adjacent offshore water and states that it does not cover the port or harbour.

**Decision: NO-GO for representing the Mombasa Port domain in its current form.**

The existing dataset names and metadata should not be interpreted as evidence of port coverage.

## 5. Vertical datum audit

### 5.1 Copernicus marine water level

The Copernicus `zos` variable is identified as:

`sea_surface_height_above_geoid`

Its vertical reference is documented as geoid-based.

This is not automatically equivalent to local mean sea level or Admiralty Chart Datum.

### 5.2 GEBCO elevation

The inspected GEBCO elevation variable declares:

- `standard_name`: `height_above_mean_sea_level`
- `long_name`: `Elevation relative to sea level`
- `units`: metres.

GEBCO documentation states that its global elevation model assumes mean sea level as the vertical reference, while acknowledging that some shallow-water source datasets may use other vertical datums.

A metadata inconsistency was identified:

- The original GEBCO file's global attribute `geospatial_bounds_vertical_crs` declares `EPSG:5831`.
- The elevation variable declares height above mean sea level.
- The file's actual `crs` variable describes horizontal WGS84 / EPSG:4326.
- The prepared Mombasa dataset repeats `bathymetry_vertical_crs: EPSG:5831`.

EPSG:5831 identifies **Instantaneous Water Level depth**, with a downward-positive depth axis. It is not a mean-sea-level height reference.

This means the EPSG:5831 metadata cannot be accepted as a valid vertical CRS declaration for the GEBCO elevation values without further clarification. The global attribute and prepared metadata should be treated as inconsistent with the elevation variable's stated reference.

No correction was made during this audit.

### 5.3 Mombasa tide-gauge zero and Chart Datum

The BODC GLOSS station record documents the Mombasa benchmark relationship:

`TGZ = Admiralty Chart Datum - 0.942 m`

where TGZ is tide-gauge zero.

For gauge observations expressed relative to TGZ, the corresponding height relative to Admiralty Chart Datum is:

`H_ACD = H_TGZ - 0.942 m`

The station record also documents relationships to the East African Railways Datum and Survey of Kenya Datum.

This provides a documented local relationship between the historical tide-gauge reference and Admiralty Chart Datum.

However, the relationship must be confirmed as applicable to the specific gauge record and period being used before operational conversion.

### 5.4 Unresolved datum transformations

The audit has not established a verified transformation connecting:

1. Copernicus geoid-referenced `zos` to local Mombasa mean sea level or Chart Datum.
2. GEBCO mean-sea-level-referenced elevations to Mombasa Chart Datum.
3. The GEBCO EPSG:5831 metadata declaration to the actual vertical reference of its elevation values.

The documented TGZ-to-Chart-Datum relationship does not solve these other transformations.

**Decision: NO-GO for combining the existing Copernicus water levels and GEBCO bathymetry into a vertically consistent operational hydrodynamic model without further datum evidence.**

No vertical offset should be invented or applied merely to make the datasets align.
