# BE58 — Historical Mombasa Hindcast Feasibility Assessment

**Project:** MSA Port Twin — Port of Mombasa  
**Scope:** Historical marine data acquisition, compatibility assessment and feasibility of constructing a Mombasa hindcast dataset  
**Model considered:** Marine FNO V2 (`fno-0.5.0`)  
**Assessment period:** Historical source-data investigation, with detailed sample analysis for October 2022  
**Status:** Feasibility assessment complete — complete hindcast dataset not produced  
**Code changes:** None  
**Model changes:** None  
**Complete hindcast dataset:** Not produced  
**Historical FNO evaluation:** Not performed

---

## 1. Executive summary

BE58 investigated whether historical atmospheric, oceanographic, tidal and observational datasets could support the construction of a historical hydrodynamic hindcast dataset for the Port of Mombasa.

Several relevant historical sources were acquired and inspected:

- ERA5 hourly 10 m wind components.
- Copernicus Marine merged hourly sea-level and ocean-tide fields.
- Copernicus GLORYS12V1 daily sea-level and current fields.
- UHSLC hourly Mombasa tide-gauge observations.
- Existing GEBCO-derived Mombasa bathymetry.

The investigation included source inspection, spatial and temporal coverage checks, data-quality checks, resolution comparisons, and preliminary comparisons between Copernicus ocean-tide estimates and UHSLC gauge observations.

The results establish that historical marine data is available for parts of the required hindcast inputs. However, the sources do not form a directly compatible, spatially and temporally consistent historical hydrodynamic dataset.

Important limitations include:

- Significant spatial-resolution differences between the source datasets and the 250 m FNO grid.
- Daily GLORYS fields that cannot directly provide hourly historical current conditions.
- Unresolved vertical-datum relationships.
- Different spatial and temporal representations of wind, tidal and oceanographic conditions.
- Differences between historical forcing and the synthetic forcing distribution used to train FNO V2.
- Limitations in the existing FNO V2 input contract.

**Overall decision: Historical hindcast reconstruction is partially feasible in principle, but a complete, defensible hindcast dataset was not produced by BE58.**

The investigation therefore closes as a documented feasibility assessment, not as a completed hindcast or historical model evaluation.

## 2. Assessment scope and evidence

The investigation covered:

1. Historical wind-data availability and compatibility.
2. Historical ocean-tide and sea-level data availability.
3. Historical current and ocean-state data availability.
4. Independent tide-gauge observations.
5. Bathymetry and spatial-domain compatibility.
6. Spatial and temporal resolution differences.
7. Compatibility with the Marine FNO V2 input contract.
8. Feasibility of constructing a consistent historical hindcast dataset.

The assessment used downloaded source samples, existing project data, dataset metadata and inspection scripts.

The principal detailed sample period was October 2022, allowing the hourly UHSLC gauge record to be compared with hourly Copernicus ocean-tide data.

The investigation did not establish a complete historical reconstruction covering the full UHSLC observation period.

## 3. Historical data sources

### 3.1 ERA5 historical wind data

**Source:** Copernicus Climate Data Store — ERA5 hourly single-level data.

**Downloaded sample:**

`data/raw/era5/mombasa_era5_wind_20221001_20221003.nc`

The sample contains:

- 10 m eastward wind component (`u10`).
- 10 m northward wind component (`v10`).
- Hourly timestamps.
- A 9 × 9 regional grid.
- Three days of data, 1–3 October 2022.
- Approximately 0.25° spatial resolution.

The downloaded fields were inspected and successfully read using the existing `ERA5WindSource`.

#### 3.1.1 Wind statistics

Statistics over the complete downloaded 9 × 9 regional grid:

| Statistic | `u10` (m/s) | `v10` (m/s) |
|---|---:|---:|
| Minimum | -6.113 | 0.643 |
| Maximum | 1.157 | 9.549 |
| Mean | -1.930 | 5.170 |
| Standard deviation | 1.197 | 1.677 |
| 5th percentile | -3.735 | 2.183 |
| 95th percentile | 0.240 | 7.830 |

Derived wind-speed statistics:

| Statistic | Wind speed (m/s) |
|---|---:|
| Minimum | 0.6754 |
| 5th percentile | 2.2706 |
| Median | 5.8400 |
| 95th percentile | 8.1123 |
| Maximum | 9.6854 |

These statistics describe the complete downloaded regional grid. They are not statistics calculated exclusively over ocean cells or the FNO domain.

#### 3.1.2 Wind-stress compatibility

The shallow-water solver applies wind forcing using:

\[
\tau_x = \rho_{air} C_d |U| U_x
\]

\[
\tau_y = \rho_{air} C_d |U| U_y
\]

The corresponding acceleration contribution is divided by water density.

The inspected solver configuration uses:

- Air density: 1.225 kg/m³.
- Water density: 1025 kg/m³.
- Wind drag coefficient: 0.0013.

Using these values, the estimated wind-stress magnitude for the downloaded ERA5 sample was:

| Statistic | Wind stress (N/m²) |
|---|---:|
| Minimum | 0.0007 |
| 5th percentile | 0.0082 |
| Median | 0.0543 |
| 95th percentile | 0.1048 |
| Maximum | 0.1494 |

The ERA5 10 m wind components are an appropriate input type for the solver's simplified wind-stress formulation.

However, the historical sample's wind-stress distribution differs from the synthetic V2 training scenarios:

- The largest synthetic scenario wind speed is 5 m/s.
- Its corresponding maximum wind stress is approximately 0.0398 N/m².
- The historical sample median stress is approximately 0.0543 N/m².
- The historical sample 95th-percentile stress is approximately 0.1048 N/m².

The historical median and 95th-percentile stresses therefore exceed the maximum synthetic scenario stress.

This indicates a forcing-distribution mismatch. It does not establish how the model would perform under historical forcing.

#### 3.1.3 Spatial-resolution limitations

The ERA5 grid is approximately 0.25° (~28 km), compared with the FNO grid spacing of 250 m.

The downloaded regional grid contains only two grid points inside the FNO domain.

ERA5 therefore provides regional atmospheric forcing, not harbour-scale wind detail.

A spatial mapping or aggregation strategy would be required before using ERA5 as input to the existing model. Mapping the coarse data onto a 250 m grid would not create additional atmospheric information.

**Decision: CONDITIONAL GO — ERA5 is a usable source of historical regional wind forcing, subject to spatial mapping and explicit treatment of the forcing-distribution mismatch.**

### 3.2 Copernicus merged hourly sea-level and ocean-tide data

**Dataset:** `cmems_mod_glo_phy_anfc_merged-sl_PT1H-i`  
**Product:** `GLOBAL_ANALYSISFORECAST_PHY_001_024`  
**Dataset version:** `202411`  
**Source attributes:** MERCATOR GLO12 and FES2014  
**Temporal resolution:** Hourly  
**Spatial resolution:** Approximately 0.083333°.

The inspected dataset contains several sea-level components, including:

- `ocean_tide`
- `sea_surface_height`
- `total_sea_level`
- `invert_barometer`
- `tide_loading`
- `global_mean_mass_volume_variation`
- `global_mean_steric_variation`

These variables represent different physical components and must not be treated as interchangeable.

In particular:

- `ocean_tide` is the explicit ocean-tide component.
- `sea_surface_height` represents ocean dynamic sea level.
- `total_sea_level` represents a combined sea-level field.

The fields must not be added together indiscriminately.

#### 3.2.1 Downloaded samples

The investigation used:

- `data/raw/copernicus/mombasa_merged_sea_level_test_20221001_20221003.nc`
- `data/raw/copernicus/mombasa_ocean_tide_202210.nc`

The short sample contains 49 hourly timestamps and a 7 × 3 spatial grid.

The October 2022 sample contains 745 timestamps because the downloaded interval includes the first timestamp of 1 November. There are 739 timestamps overlapping the UHSLC gauge record.

The nearest inspected gauge comparison grid cell is:

- Latitude: -4.0830078125°
- Longitude: 39.75°

The longitude separation from the gauge is approximately 10 km.

#### 3.2.2 Comparison with UHSLC observations

October 2022 gauge observations were compared with the Copernicus `ocean_tide` component at the nearest grid cell.

The comparison used demeaned time series to examine tidal variation without treating the two datasets' absolute reference levels as aligned.

| Metric | Result |
|---|---:|
| Overlapping hourly observations | 739 |
| Correlation | 0.9626 |
| Copernicus standard deviation | 0.8558 m |
| Gauge standard deviation | 0.8690 m |
| Copernicus range | 3.655 m |
| Gauge range | 3.751 m |
| Demeaned RMSE | 0.2360 m |

A wider comparison across 20 valid Copernicus grid cells produced:

- Correlations between approximately 0.962 and 0.963.
- Median correlation of approximately 0.962.
- Median demeaned RMSE of approximately 0.238 m.

This is preliminary evidence that the Copernicus ocean-tide component reproduces the broad tidal timing and variation observed at the Mombasa gauge during the sampled month.

It is not independent validation across 20 stations: all 20 grid cells were compared against the same gauge.

#### 3.2.3 Limitations

The comparison is limited by:

- One month of detailed comparison.
- One tide-gauge station.
- Approximately 9 km source-grid resolution.
- Point-to-grid spatial separation.
- Unresolved absolute vertical-datum alignment.
- The combined GLO12/FES2014 source provenance.

The comparison supports further investigation of this dataset as a regional tidal-forcing source. It does not establish operational suitability or a fully validated local tidal boundary condition.

**Decision: CONDITIONAL GO — Copernicus `ocean_tide` is a promising historical tidal-data source, but datum alignment, spatial limitations and broader validation remain necessary.**

### 3.3 UHSLC Mombasa tide-gauge observations

**Source:** University of Hawaiʻi Sea Level Center (UHSLC), Research Quality Data.

**Raw file:**

`data/raw/observations/mombasa_uhslc_hourly_2020_2024.csv`

The record contains:

- Station ID: 101.
- GLOSS ID: 8.
- Period: 1 January 2020 to 31 December 2024.
- Hourly sampling.
- Latitude: -4.06999°.
- Longitude: 39.65732°.
- Measurement units: millimetres.
- Reference datum: station zero.
- Quality flag: 4 for all inspected rows.

The CSV contains 43,848 hourly rows:

| Check | Result |
|---|---:|
| Data rows | 43,848 |
| Expected hourly timestamps | 43,848 |
| Duplicate timestamps | 0 |
| Timestamp gaps | 0 |
| Finite observations | 39,021 |
| Missing/non-finite observations | 4,827 |
| Finite observation proportion | Approximately 89.0% |

The gauge provides an independent historical water-level observation record.

It does not provide spatially distributed initial water levels or current fields.

#### 3.3.1 Datum limitation

The gauge measurements use station zero as their reference datum.

The BE52 audit documents a BODC benchmark relationship involving Admiralty Chart Datum and the tide-gauge zero. However, a datum conversion must not be applied automatically without confirming its applicability and sign convention against the relevant benchmark documentation.

The FNO V2 water-level reference and the GEBCO-derived bathymetric reference have not been fully reconciled with the gauge reference.

Demeaned comparisons can examine tidal variation while avoiding a direct comparison of absolute levels, but they do not resolve the underlying datum difference.

**Decision: GO — The UHSLC record is suitable as an independent historical water-level observation source, subject to quality filtering and explicit datum limitations.**

### 3.4 GLORYS12V1 historical ocean reanalysis

**Dataset:** `cmems_mod_glo_phy_my_0.083deg_P1D-m`  
**Dataset version:** `202311`  
**Product:** GLORYS12V1  
**Temporal resolution:** Daily mean  
**Spatial resolution:** Approximately 0.083° (~8 km)  
**Vertical resolution:** 50 levels  
**Coverage:** 1993 onwards.

The inspected fields include:

- `zos`: sea-surface height above the geoid.
- `uo`: eastward ocean current.
- `vo`: northward ocean current.

The dataset assimilates altimetry and in-situ temperature/salinity observations.

#### 3.4.1 Downloaded sample

`data/raw/copernicus/mombasa_glorys_hindcast_test_20221001_20221003.nc`

The sample contains:

- Three daily timestamps: 1–3 October 2022.
- A 25 × 25 regional latitude/longitude grid.
- 50 vertical levels.
- Approximately 0.083° spatial spacing.
- Surface depth level at 0.494025 m.

At the surface, each of the inspected variables has 411 valid cells out of 625, approximately 65.8%.

The sample demonstrates that historical regional sea-level and current fields are available for the selected period.

#### 3.4.2 Comparison with the FNO domain

The existing FNO domain uses a projected 250 m grid:

- Dimensions: 131 × 199.
- CRS: EPSG:32737.
- Easting: 561,500–594,000 m.
- Northing: 9,521,250–9,570,750 m.

The GLORYS source grid is substantially coarser.

A nearest-gridpoint diagnostic over the 15,562 GEBCO-covered FNO cells produced:

| Distance statistic | Distance |
|---|---:|
| Median | 3,587 m |
| 75th percentile | 4,464 m |
| 90th percentile | 5,097 m |
| 95th percentile | 5,490 m |
| 99th percentile | 6,048 m |
| Maximum | 6,435 m |

For each of the three sampled dates, valid nearest source points were found for 15,304 of the 15,562 FNO cells, approximately 98.3%.

However, only 18 distinct GLORYS grid cells supplied values to those 15,304 FNO cells.

The median number of FNO cells mapped to one GLORYS cell was 912.5, with a maximum of 1,369.

This means that thousands of 250 m cells would inherit the same coarse source values under nearest-neighbour mapping.

It would be incorrect to interpret the resulting 250 m grid as containing 250 m historical oceanographic information.

#### 3.4.3 Temporal and physical limitations

GLORYS provides daily means, not hourly tidal currents.

Its `zos` field is sea-surface height above the geoid. It is not an astronomical tide component and must not be substituted for or indiscriminately added to Copernicus `ocean_tide`.

Daily GLORYS currents can describe broad regional ocean conditions, but they cannot directly reconstruct the hourly current variations required for a six-hour hourly forecast comparison.

**Decision: CONDITIONAL GO — GLORYS is useful for broad historical regional ocean-state context, but its spatial and daily temporal resolution do not directly support harbour-scale hourly hydrodynamic initial conditions.**

### 3.5 GEBCO-derived bathymetry

The existing projected bathymetry source is:

`data/processed/marine/bathymetry/projected/approaches_gebco_2026_utm37s_250m.nc`

The grid uses:

- CRS: EPSG:32737.
- Grid spacing: 250 m.
- Dimensions: 199 northing × 131 easting.
- Positive-down bathymetry.
- Source vertical reference described as height above mean sea level.

The inspected `bathymetry_m`, `tid` and source-coverage fields identify 15,562 cells with source coverage. Another 10,507 cells do not have source coverage.

The source-coverage mask is not a navigability mask and does not establish that every covered cell is safe or suitable for vessel operations.

The existing BE52 audit also identified limitations in the underlying GEBCO resolution for detailed berth and channel applications.

The bathymetry is available as a spatial model input, but its vertical reference has not been reconciled fully with the gauge and historical sea-level references.

**Decision: CONDITIONAL GO — Existing GEBCO-derived bathymetry can provide the current model-grid bathymetric input, subject to coverage and vertical-datum limitations. It is not independently verified channel-scale hydrographic survey data.**

## 4. Historical source compatibility

The source datasets differ substantially in spatial resolution, temporal resolution and physical meaning.

| Source | Main variables | Temporal resolution | Approximate spatial resolution | Main limitation |
|---|---|---|---|---|
| ERA5 | `u10`, `v10` | Hourly | 0.25° (~28 km) | Coarse regional wind forcing |
| Copernicus merged sea level | `ocean_tide`, sea-level components | Hourly | 0.083° (~9 km) | Datum and local boundary limitations |
| GLORYS12V1 | `zos`, `uo`, `vo` | Daily | 0.083° (~8 km) | Daily averages and coarse spatial coverage |
| UHSLC | Sea level | Hourly | Single station | Point observation; no spatial currents |
| GEBCO | Bathymetry | Static product | Source-dependent | Resolution and vertical-datum limitations |

These datasets cannot be treated as interchangeable observations of the same physical quantity.

A complete reconstruction would require, at minimum:

1. Consistent historical timestamps and forecast reference times.
2. Spatial mapping onto a common model domain.
3. Explicit handling of missing values and land masks.
4. A justified temporal treatment of daily GLORYS fields.
5. A defined method for extracting or constructing tidal boundary conditions.
6. Verified vertical-datum relationships.
7. Consistent provenance and source metadata.

These preparation and reconciliation steps were not completed as a unified hindcast-data pipeline in BE58.

## 5. Compatibility with Marine FNO V2

### 5.1 Existing model contract

The inspected FNO V2 contract is:

| Property | Value |
|---|---|
| Model version | `fno-0.5.0` |
| Dataset version | `v2` |
| Input contract | `marine-fno-v2-1.0` |
| Input channels | 13 |
| Output channels | 3 |
| Forecast horizon | 6 hourly steps |
| Output shape | `[batch, 6, 3, ny, nx]` |

The 13 input channels are:

1. Bathymetry.
2. Initial water level.
3. Initial eastward current.
4. Initial northward current.
5. Water mask.
6. Normalized easting.
7. Normalized northing.
8. Eastward wind component.
9. Northward wind component.
10. Tidal amplitude.
11. Tidal phase sine.
12. Tidal phase cosine.
13. Eastern boundary indicator.

The model outputs water level, eastward current and northward current.

### 5.2 Historical input compatibility

The existing contract can represent:

- Spatial initial water-level fields.
- Spatial initial eastward and northward current fields.
- Bathymetry and water mask.
- Spatially uniform wind components.
- A spatially uniform tidal amplitude and phase.
- A fixed eastern tidal boundary indicator.

However, it does not directly represent:

- Spatially varying wind fields.
- Time-varying wind forcing during the six-hour forecast.
- Spatially varying tidal boundary conditions.
- A directly supplied time series of historical boundary elevation.
- Arbitrary boundary locations or boundary geometries.

ERA5 provides spatially varying hourly wind fields, while the current V2 contract broadcasts scalar wind components uniformly across the grid.

Copernicus `ocean_tide` provides a spatially varying hourly tidal field, while the V2 contract uses scalar amplitude and phase with a fixed eastern boundary.

Converting these historical fields to scalar inputs would discard spatial and temporal information.

### 5.3 Synthetic training distribution

The V2 training dataset was generated from 18 synthetic forcing combinations:

- Six wind profiles.
- Three tidal profiles.

The wind profiles include:

- Calm.
- Eastward 3 m/s.
- Westward 3 m/s.
- Northward 3 m/s.
- Southward 3 m/s.
- Eastward 5 m/s.

The tidal profiles use amplitudes of:

- 0.35 m.
- 0.50 m.
- 0.80 m.

The phases are paired with those amplitudes at:

- 0 radians.
- π/3 radians.
- 2π/3 radians.

Because amplitude and phase are paired in this scenario design, their effects are not independently varied across the complete set of profiles.

The dataset contains 126 six-hour windows extracted from the 18 scenarios:

| Split | Scenarios | Windows |
|---|---:|---:|
| Training | 12 | 84 |
| Validation | 3 | 21 |
| Test | 3 | 21 |
| **Total** | **18** | **126** |

The scenario-isolated split avoids placing windows from the same scenario into different splits. However, the 126 windows are overlapping windows from only 18 independent simulated forcing combinations.

All scenarios share the same model domain, bathymetry, water mask and solver. Initial conditions at the beginning of each simulation are zero water-level anomaly and zero currents.

The dataset therefore represents a limited synthetic simulation distribution, not the range of observed historical Mombasa conditions.

### 5.4 Implications for historical evaluation

The historical sources inspected in BE58 do not directly provide a complete set of model-compatible initial and boundary conditions.

Even if approximate inputs were constructed, they would differ from the synthetic training distribution in important ways:

- Historical initial water levels and currents would generally be non-zero.
- Historical winds have a different magnitude and spatial distribution.
- Historical tidal conditions vary spatially and over time.
- GLORYS currents are daily averages rather than hourly conditions.
- The historical vertical references have not been fully reconciled.

Consequently, running the existing FNO V2 on approximated historical inputs would involve substantial distribution shift.

A resulting prediction could be exploratory, but it would not establish historical forecast skill or real-world model accuracy.

**Decision: NO-GO — The existing FNO V2 pipeline does not currently support a scientifically defensible historical hindcast evaluation using the acquired data without additional input reconstruction and model-readiness work.**

## 6. Consolidated feasibility decisions

| Assessment area | Decision | Basis |
|---|---|---|
| Historical wind acquisition | GO | ERA5 hourly wind sample obtained and inspected |
| Historical ocean-tide acquisition | GO | Copernicus hourly `ocean_tide` sample obtained |
| Independent water-level observations | GO | UHSLC Mombasa gauge record obtained and inspected |
| Historical regional currents | CONDITIONAL GO | GLORYS provides daily regional current fields |
| Existing bathymetry availability | CONDITIONAL GO | Projected GEBCO-derived grid exists, with coverage and datum limitations |
| Historical tidal comparison | CONDITIONAL GO | Preliminary October 2022 comparison with UHSLC shows strong agreement in demeaned variation |
| Unified historical input reconstruction | NO-GO for BE58 completion | Sources were not fully aligned and assembled into a single dataset |
| Harbour-scale historical current reconstruction | NO-GO with current evidence | GLORYS resolution and daily sampling are insufficient |
| Direct historical FNO V2 evaluation | NO-GO | Missing compatible inputs and substantial training-distribution differences |
| Complete historical Mombasa hindcast dataset | NOT PRODUCED | No unified model-ready hindcast dataset was assembled |

## 7. Overall recommendation

BE58 establishes that relevant historical data sources exist and that some components are promising for further reconstruction work.

The strongest evidence is:

- ERA5 provides hourly historical regional wind forcing.
- Copernicus merged `ocean_tide` provides hourly tidal variation with a promising preliminary comparison against the Mombasa gauge.
- UHSLC provides an independent hourly water-level observation record.
- GLORYS provides historical regional ocean-state information.
- Existing GEBCO-derived bathymetry provides a model-grid bathymetric input, subject to its documented limitations.

However, the sources are not yet a consistent historical hydrodynamic dataset.

The remaining issues are not simply file-format or interpolation tasks. They include physical compatibility, temporal resolution, spatial resolution, datum alignment, boundary-condition reconstruction and model training-distribution limitations.

**Final BE58 outcome: Historical data acquisition and feasibility assessment completed. A complete historical Mombasa hindcast dataset was not produced, and no historical FNO V2 evaluation was performed.**

No model retraining, architecture changes, inference changes or production integration are recommended as part of closing this assessment.

Any future hindcast effort should be treated as a separate, explicitly scoped task with defined historical periods, input requirements, datum treatment, spatial and temporal reconstruction methods, and independent validation criteria.

## 8. Evidence limitations

The conclusions in this assessment are subject to the following limitations:

1. ERA5 wind statistics describe the complete downloaded 9 × 9 regional sample, not only ocean cells within the FNO domain.
2. The ERA5 sample covers three days and does not establish the distribution of winds over the full historical gauge period.
3. The Copernicus–UHSLC tidal comparison covers October 2022 and one gauge station.
4. The 20 Copernicus grid-cell comparisons use the same gauge record and are not 20 independent station validations.
5. Absolute sea-level datum alignment remains unresolved.
6. GLORYS inspection covers a three-day sample and does not establish suitability for every historical period.
7. Nearest-gridpoint mapping diagnostics do not establish the accuracy of interpolated fields or their physical suitability as model initial conditions.
8. Regridding coarse source data to 250 m does not create new high-resolution observations.
9. The FNO V2 training scenarios are synthetic and do not establish historical forecast skill.
10. No complete synchronized historical input dataset or historical model evaluation was produced.

The assessment should not be interpreted as proof that a historical hindcast is impossible. It establishes that the available evidence and existing model pipeline were insufficient to complete one defensibly within BE58.

## 9. Data and reproducibility notes

The following source files were inspected during the investigation:

| File | Description |
|---|---|
| `data/raw/era5/mombasa_era5_wind_20221001_20221003.nc` | ERA5 hourly wind sample |
| `data/raw/copernicus/mombasa_merged_sea_level_test_20221001_20221003.nc` | Short Copernicus merged sea-level sample |
| `data/raw/copernicus/mombasa_ocean_tide_202210.nc` | October 2022 hourly Copernicus ocean-tide sample |
| `data/raw/copernicus/mombasa_glorys_hindcast_test_20221001_20221003.nc` | GLORYS daily ocean-state sample |
| `data/raw/observations/mombasa_uhslc_hourly_2020_2024.csv` | UHSLC hourly gauge observations |
| `data/processed/marine/bathymetry/projected/approaches_gebco_2026_utm37s_250m.nc` | Existing projected GEBCO-derived bathymetry |

These files are source-data samples and supporting project data. Their presence does not mean a unified hindcast dataset was created.

Large external NetCDF downloads should not be committed to the repository by default. Preserve acquisition details, source identifiers, metadata and reproducible inspection procedures. Any future decision to commit small test fixtures should be made deliberately.

## 10. References

1. Copernicus Climate Data Store — [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/).
2. Copernicus Marine Service — [Marine Data Store](https://data.marine.copernicus.eu/).
3. University of Hawaiʻi Sea Level Center — [Research Quality Data](https://uhslc.soest.hawaii.edu/erddap/).
4. BODC / GLOSS — [Mombasa Station Information Sheet](https://www.bodc.ac.uk/resources/inventories/gloss_handbook/stations/8/).
5. BODC — [Mombasa Site History](https://www.bodc.ac.uk/data/documents/nodb/57529/).
6. GEBCO — [General Bathymetric Chart of the Oceans](https://www.gebco.net/).
7. BE52 — `docs/research/BE52-marine-data-audit.md`, documenting marine-data provenance, resolution, tide and vertical-datum limitations.
8. BE57 — `docs/research/BE57-marine-model-validation.md`, documenting the limitations of independent real-world FNO V2 validation.
9. Project implementation — `port_twin_app/services/marine_ai/model_v2.py`, documenting the FNO V2 model and input contract.
10. Project implementation — `port_twin_app/services/marine_ai/training_v2.py`, documenting V2 input construction and training-data preparation.
11. Project implementation — `port_twin_app/services/marine_ai/preparation_v2.py`, documenting V2 scenario and forcing requirements.
12. Project implementation — `port_twin_app/services/marine_data/shallow_water/solver.py`, documenting the shallow-water solver and wind-stress formulation.

