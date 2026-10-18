# Licenses and attribution

## Analysis code

The analysis code and command-line tools are licensed under the [MIT license](LICENSE):

> Copyright (c) 2026 The Authors

## Earthquake catalogs

| File | Attribution and license |
| --- | --- |
| `data/catalog.txt` | Fountoulakis and Evangelidis, version 2.1, [Zenodo 15240096](https://doi.org/10.5281/zenodo.15240096); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `data/Santorini_catalog.csv` | Karakostas et al., version 1.0, [Zenodo 15111649](https://doi.org/10.5281/zenodo.15111649); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The source files are unmodified. The generated `data/santorini_e_catalog.csv` is a format conversion and retains the first catalog's attribution and license. Full dataset citations appear in [DATA_SOURCES.md](DATA_SOURCES.md).

## Map files

| File | Attribution and terms |
| --- | --- |
| `maps/NOAFaults_v6.0.xy` | Fault traces from NOAFAULTS version 6.0 by A. Ganas / National Observatory of Athens, [Zenodo 13168947](https://doi.org/10.5281/zenodo.13168947); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| `maps/Santorini_NE_2025.xy` | GMT-style coastline coordinates. The upstream dataset, version, and redistribution license are not specified. |
| `maps/Kolumbo.xyz` | Coordinates accompanied by a Nomikou et al. (2016), Tectonophysics 671, 202–217 reference; no separate license notice. |
| `maps/lomax2025_source_traces.xy` | Approximate modeled-source context after the figure in Lomax et al. (2025), Science, [DOI: 10.1126/science.adz8538](https://doi.org/10.1126/science.adz8538). Source-figure permissions and redistribution terms are not specified. |
| `maps/ne10m_bathy_contours.xy` | Attributed to the public-domain Natural Earth 1:10 million-scale 200 m bathymetric boundary. Source layer and interpretation are described in `DATA_SOURCES.md`. |

The code's MIT license does not extend to source datasets or map layers. Figures that incorporate these materials retain the relevant attribution and source terms.
