# Data sources

## Earthquake catalogs

### SantoriniE — `data/catalog.txt`

I. Fountoulakis and C. P. Evangelidis, *Supplementary Dataset for the Report “The 2024–2025 seismic sequence in the Santorini-Amorgos region: Insights into volcano-tectonic activity through high-resolution seismic monitoring”*, version 2.1, Zenodo, 2025. [DOI: 10.5281/zenodo.15240096](https://doi.org/10.5281/zenodo.15240096). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**SantoriniE** is the label used in this analysis. The source file has a pipe-delimited header and whitespace-delimited event rows. `data/prepare_santorini_e_catalog.py` creates `data/santorini_e_catalog.csv`, preserving the source columns and values in CSV format. The conversion retains the catalog's source attribution and license.

### NLL — `data/Santorini_catalog.csv`

V. Karakostas et al., *2025 Santorini–Amorgos NLL-SSST-coherence high-precision relocated earthquake catalog*, version 1.0, Zenodo, 2025. [DOI: 10.5281/zenodo.15111649](https://doi.org/10.5281/zenodo.15111649). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This relocated comparison catalog is labeled **NLL** in the configuration and results and **NLL-SSST** in the figures. The analysis reads the source CSV directly.

Both source catalogs are unmodified release files. Their SHA-256 checksums are listed in `checksums.sha256`.

## Map context

These files provide visual context in `fig_map_overlay.png`; they do not enter the numerical seismicity analyses.

### Active faults — `maps/NOAFaults_v6.0.xy`

A. Ganas, *NOAFAULTS KMZ layer Version 6.0*, National Observatory of Athens, 2024. [DOI: 10.5281/zenodo.13168947](https://doi.org/10.5281/zenodo.13168947). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Longitude/latitude fault traces from the cited NOAFAULTS layer provide regional tectonic context and retain the source attribution and CC BY 4.0 license.

### Coastline — `maps/Santorini_NE_2025.xy`

Longitude/latitude coastline context in GMT format. The upstream dataset, version, and redistribution license are not specified.

### Kolumbo marker — `maps/Kolumbo.xyz`

Marker position: **36.52304° N, 25.48632° E**. The map code uses these coordinates to place Kolumbo.

The accompanying reference is **Nomikou et al. (2016), Tectonophysics 671, 202–217**. The coordinate file has no separate license notice.

### Modeled-source traces — `maps/lomax2025_source_traces.xy`

Approximate modeled-source traces after the modeled-dike figure in **Lomax et al. (2025), Science**, [DOI: 10.1126/science.adz8538](https://doi.org/10.1126/science.adz8538).

These traces provide qualitative map context. They are not an official data release or numerical inputs to the seismicity analysis. Source-figure permissions and redistribution terms are not specified.

### Bathymetric boundary — `maps/ne10m_bathy_contours.xy`

The file is attributed to the **200 m depth polygon boundary** in the Natural Earth 1:10 million-scale layer `ne_10m_bathymetry_K_200`, a cartographic generalization derived from SRTM30_PLUS. Source: [Natural Earth vector repository](https://github.com/nvkelso/natural-earth-vector). Natural Earth data are public domain.

This generalized boundary provides regional seafloor context; it is not a detailed local bathymetric survey.

## Attribution

Cite the dataset and publication records above when reusing the catalogs, map layers, or derived figures. See [LICENSES.md](LICENSES.md) for the terms associated with each file group.
