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
