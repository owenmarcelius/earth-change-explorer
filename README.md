# Earth Change Explorer
NASA Space Apps 2026 project. Select an Earth science variable and region to see
**what** is changing, **where**, **how much**, and whether it is **statistically significant**.

**Live app:** (paste your streamlit.app link here)

## Data
**NASA:** MODIS/Terra Vegetation Indices Monthly L3 (MOD13C2 v061, NASA LP DAAC); GISS Surface Temperature Analysis (GISTEMP v4, NASA GISS).
**Other open data:** NOAA ERSSTv5, GPCP v2.3 precipitation, NOAA interpolated OLR, NCEP/NCAR Reanalysis 1 (NOAA/NCAR), COBE-SST2 sea ice (JMA, via NOAA PSL), Natural Earth coastlines.

## AI use
The application code and data-processing scripts were generated with the AI assistant Claude (Anthropic),
then run, tested and edited by the team. No AI-generated images or video are used.

## Software
Python, Streamlit, xarray, NumPy, pandas, SciPy, pymannkendall, Plotly, earthaccess, pyhdf,
Cloudflare Tunnel, Google Colab. Fonts: Orbitron, Rajdhani, Share Tech Mono (Google Fonts).

## Method
Seasonal/annual means, per-cell linear regression for maps, area-weighted regional series
tested with Mann-Kendall + Sen's slope (with autocorrelation-corrected options).
