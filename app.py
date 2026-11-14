# Earth Change Explorer, NASA Space Apps 2026.
# Written with AI assistance (Claude, Anthropic); run, tested and edited by the team. See README.
import os, json
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import pymannkendall as mk
from scipy import stats

st.set_page_config(page_title="Earth Change Explorer", page_icon="🛰️", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"], .stMarkdown, p, label { font-family: 'Rajdhani', sans-serif !important; font-size: 1.05rem; }

.stApp {
  background:
    radial-gradient(1px 1px at 20px 30px, #fff, transparent),
    radial-gradient(1.5px 1.5px at 90px 140px, #cfe3ff, transparent),
    radial-gradient(1px 1px at 160px 70px, #fff, transparent),
    radial-gradient(2px 2px at 230px 200px, #ffe9b0, transparent),
    radial-gradient(1px 1px at 300px 120px, #fff, transparent),
    radial-gradient(1px 1px at 60px 220px, #9fc2ff, transparent),
    radial-gradient(ellipse at 15% 0%, #16225a 0%, #070b1f 45%, #02030a 100%);
  background-size: 340px 260px, 340px 260px, 340px 260px, 340px 260px, 340px 260px, 340px 260px, 100% 100%;
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1300px; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1030 0%, #050816 100%);
  border-right: 1px solid rgba(79,195,247,.25);
}
[data-testid="stSidebar"] h2 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .08em; font-size: 1.1rem; color: #4fc3f7; }

.hero { position: relative; overflow: hidden; padding: 30px 36px; border-radius: 18px; margin-bottom: 20px;
  background: linear-gradient(120deg, rgba(10,20,70,.85) 0%, rgba(8,14,45,.85) 60%, rgba(0,60,80,.7) 100%);
  border: 1px solid rgba(79,195,247,.35);
  box-shadow: 0 0 30px rgba(79,195,247,.15), inset 0 0 60px rgba(40,80,200,.12); }
.hero .tag { font-family: 'Share Tech Mono', monospace; color: #7dffb2; font-size: .8rem; letter-spacing: .18em; }
.hero h1 { font-family: 'Orbitron', sans-serif !important; font-weight: 900; font-size: 2.2rem; letter-spacing: .06em;
  margin: 8px 0 6px 0; padding: 0;
  background: linear-gradient(90deg, #ffffff, #4fc3f7 60%, #7dffb2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 0; max-width: 70%; color: #b9ccf5; font-size: 1.1rem; }
.hero .planet { position: absolute; right: -40px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle at 32% 30%, #6fe3ff 0%, #1e6fd9 35%, #0a2a7a 65%, #020618 100%);
  box-shadow: 0 0 60px rgba(79,195,247,.55), inset -30px -20px 60px rgba(0,0,0,.7); animation: float 7s ease-in-out infinite alternate; }
.hero .ring { position: absolute; right: -90px; top: 40px; width: 380px; height: 70px; border-radius: 50%;
  border: 2px solid rgba(255,214,120,.55); transform: rotate(-18deg); box-shadow: 0 0 14px rgba(255,214,120,.35); }
@keyframes float { from { transform: translateY(0); } to { transform: translateY(14px); } }

.kpi { background: rgba(14,22,60,.65); backdrop-filter: blur(6px); border-radius: 14px; padding: 14px 18px;
  border: 1px solid rgba(120,160,255,.2); }
.kpi .l { font-family: 'Share Tech Mono', monospace; font-size: .75rem; text-transform: uppercase; letter-spacing: .15em; color: #7f93c4; }
.kpi .v { font-family: 'Orbitron', sans-serif; font-size: 1.45rem; font-weight: 700; color: #ffffff; line-height: 1.35; }
.kpi .s { font-size: .85rem; color: #8ea3d6; }

.callout { background: rgba(14,22,60,.65); backdrop-filter: blur(6px); border-left: 5px solid #4fc3f7; border-radius: 10px;
  padding: 16px 20px; color: #dbe7ff; font-size: 1.1rem; border-top: 1px solid rgba(120,160,255,.15);
  border-right: 1px solid rgba(120,160,255,.15); border-bottom: 1px solid rgba(120,160,255,.15); }

.pill { display:inline-block; padding: 4px 14px; border-radius: 999px; font-size: .9rem; font-weight: 600; letter-spacing: .02em; }
.pill.up { background: rgba(255,82,82,.15); color: #ff8a80; border: 1px solid rgba(255,82,82,.5); }
.pill.down { background: rgba(79,195,247,.15); color: #80d8ff; border: 1px solid rgba(79,195,247,.5); }
.pill.ok { background: rgba(125,255,178,.12); color: #7dffb2; border: 1px solid rgba(125,255,178,.4); }
.pill.no { background: rgba(255,255,255,.07); color: #b0b7c3; border: 1px solid rgba(255,255,255,.18); }

button[data-baseweb="tab"] { font-family: 'Orbitron', sans-serif !important; font-size: .8rem !important; letter-spacing: .06em; color: #b9ccf5 !important; }
button[data-baseweb="tab"] p { color: inherit !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #4fc3f7 !important; }
div[data-baseweb="tab-highlight"] { background-color: #4fc3f7 !important; box-shadow: 0 0 10px #4fc3f7; }

h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .04em; color: #cfe3ff; }

/* sidebar text visible */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #dbe7ff !important; }
[data-testid="stSidebar"] label[data-baseweb="radio"] div { color: #dbe7ff !important; }
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
[data-testid="stSidebar"] [data-testid="stThumbValue"] { color: #7dd3fc !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] div { color: #1b2430 !important; }

/* case-study buttons */
.stButton > button { background: rgba(14,22,60,.7); border: 1px solid rgba(79,195,247,.45); border-radius: 12px;
  font-family: 'Rajdhani', sans-serif; font-weight: 700; transition: all .15s; }
.stButton > button:hover { border-color: #7dffb2; box-shadow: 0 0 14px rgba(125,255,178,.35); }
[data-testid="stMain"] .stButton button p { color: #dbe7ff !important; }
[data-testid="stMain"] .stButton button:hover p { color: #7dffb2 !important; }

/* main page text visible */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: #dbe7ff !important; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4 { color: #cfe3ff !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

REC = st.session_state.get("rec_mode", False)   # recording layout: hide banner + top cards

HERO_HTML = """
<div class="hero">
<div class="planet"></div><div class="ring"></div>
<div class="tag">● SYSTEMS NOMINAL &nbsp;·&nbsp; NASA SPACE APPS &nbsp;·&nbsp; EARTH OBSERVATION</div>
<h1>EARTH CHANGE EXPLORER</h1>
<p>Pick a variable and a region to find out <b>what</b> is changing, <b>where</b>,
<b>how much</b>, and whether it is <b>statistically significant</b>.</p>
</div>
"""
if not REC:
    st.markdown(HERO_HTML, unsafe_allow_html=True)

pio.templates["space"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,14,40,0.55)",
    font=dict(color="#dbe7ff", family="Rajdhani, sans-serif", size=14),
    xaxis=dict(gridcolor="rgba(120,150,255,.12)", zerolinecolor="rgba(120,150,255,.3)"),
    yaxis=dict(gridcolor="rgba(120,150,255,.12)", zerolinecolor="rgba(120,150,255,.3)"),
    colorway=["#4fc3f7", "#ff8a65", "#7dffb2", "#ffd54f"],
    legend=dict(bgcolor="rgba(0,0,0,0)")))

def show(fig):
    """Draw a 2D chart with the dark theme."""
    fig.update_layout(template="space", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(8,14,40,0.55)", font=dict(color="#dbe7ff"))
    fig.update_xaxes(gridcolor="rgba(120,150,255,.12)", linecolor="rgba(120,150,255,.3)")
    fig.update_yaxes(gridcolor="rgba(120,150,255,.12)", linecolor="rgba(120,150,255,.3)")
    st.plotly_chart(fig, use_container_width=True, theme=None)

def plain(label):
    """Menu labels start with an emoji; drop it."""
    return label.split(" ", 1)[1]

def kpi(label, value, sub="", accent="#4fc3f7"):
    return (f'<div class="kpi" style="border-left:4px solid {accent}; box-shadow: 0 0 18px {accent}33">'
            f'<div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')

VARIABLES = {
    "🌡️ Surface temperature anomaly": dict(
        file="gistemp.nc", var="tempanomaly", units="°C", scale=1, maxabs=100, cmap="RdBu_r",
        source="NASA GISTEMP v4", info="Land + ocean temperature relative to 1951–1980."),
    "🌊 Sea surface temperature": dict(
        file="ersst.nc", var="sst", units="°C", scale=1, maxabs=1e10, cmap="RdBu_r",
        source="NOAA ERSSTv5", info="Monthly sea surface temperature, 2° grid."),
    "🌧️ Precipitation": dict(
        file="gpcp.nc", var="precip", units="mm/day", scale=1, maxabs=1e10, cmap="BrBG",
        source="GPCP v2.3 (satellite + gauges)", info="Blue/green = wetter, brown = drier."),
    "☁️ Outgoing longwave radiation": dict(
        file="olr.nc", var="olr", units="W/m²", scale=1, maxabs=1e10, cmap="RdBu_r",
        source="NOAA interpolated OLR", info="Lower OLR usually means more high cloud / convection."),
    "🧭 Sea level pressure": dict(
        file="slp.nc", var="slp", units="hPa", scale=0.01, maxabs=1e10, cmap="RdBu_r",
        source="NCEP/NCAR reanalysis (model)", info="Shows shifts in atmospheric circulation."),
    "💧 Precipitable water vapor": dict(
        file="prwtr.nc", var="pr_wtr", units="kg/m²", scale=1, maxabs=1e10, cmap="BrBG",
        source="NCEP/NCAR reanalysis (model)", info="Total water vapor in the air column."),
    "🧊 Sea ice concentration": dict(
        file="icec.nc", var="icec", units="%", scale=100, maxabs=1e10, cmap="RdBu",
        source="COBE-SST2 (JMA, via NOAA PSL)",
        info="Red = ice loss, blue = ice gain. Use the Arctic or Southern Ocean presets."),
    "🌿 Vegetation greenness (NDVI)": dict(
        file="ndvi.nc", var="ndvi", units="NDVI", scale=1, maxabs=10, cmap="BrBG",
        source="NASA MODIS Terra MOD13C2 v061 (aggregated)",
        info="Green = greening, brown = browning. 2001 onward."),
}
for v in VARIABLES.values():
    v["file"] = os.path.join("data", v["file"])
VARIABLES = {k: v for k, v in VARIABLES.items() if os.path.exists(v["file"])}
if not VARIABLES:
    st.error("No data files found in the data/ folder. Run the data cells first."); st.stop()

REGIONS = {  # lat0, lat1, lon0, lon1
    "🌐 Global": (-90, 90, -180, 180),
    "🧊 Arctic (66–90°N)": (66, 90, -180, 180),
    "🐧 Antarctic (60–90°S)": (-90, -60, -180, 180),
    "☀️ Tropics (23.5°S–23.5°N)": (-23.5, 23.5, -180, 180),
    "🌴 Maritime Continent / Indonesia": (-11, 6, 95, 141),
    "🌿 Amazon basin": (-15, 5, -75, -50),
    "🏜️ Sahel": (10, 18, -18, 40),
    "🌾 South Asian monsoon": (8, 30, 68, 92),
    "🇪🇺 Europe": (36, 60, -10, 30),
    "🇺🇸 Western USA": (31, 49, -125, -105),
    "🌊 Niño 3.4 (tropical Pacific)": (-5, 5, -170, -120),
    "🌀 Southern Ocean": (-65, -45, -180, 180),
    "✏️ Custom (use sliders)": None,
}
SEASONS = {"Annual": [1,2,3,4,5,6,7,8,9,10,11,12], "Dec–Feb": [12,1,2],
           "Mar–May": [3,4,5], "Jun–Aug": [6,7,8], "Sep–Nov": [9,10,11]}
TESTS = {"Mann-Kendall (standard)": mk.original_test,
         "Hamed-Rao (corrects autocorrelation)": mk.hamed_rao_modification_test,
         "Yue-Wang (corrects autocorrelation)": mk.yue_wang_modification_test}

def open_clean(cfg):
    ds = xr.open_dataset(cfg["file"], decode_times=False)
    da = ds[cfg["var"]]

    units = ds["time"].attrs.get("units", "days since 1800-01-01")
    step, origin = units.split(" since ")
    origin = pd.Timestamp(origin.split()[0])
    unit = {"days": "D", "hours": "h", "seconds": "s"}.get(step.strip().lower(), "D")

    t = ds["time"].values.astype("float64")
    good = np.isfinite(t) & (np.abs(t) < 1e9)
    da = da.isel(time=good)
    times = origin + pd.to_timedelta(t[good], unit=unit)
    da = da.assign_coords(time=times)
    da = da.sel(time=~pd.Index(da.time.values).duplicated())

    for d in list(da.dims):
        if d not in ("time", "lat", "lon"):
            da = da.isel({d: 0}, drop=True)
    da = da.where(np.abs(da) < cfg["maxabs"]) * cfg["scale"]
    da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")
    return da

@st.cache_data(show_spinner="Reading data…")
def load_seasonal(name, season):
    da = open_clean(VARIABLES[name])
    months = SEASONS[season]
    sub = da.sel(time=da.time.dt.month.isin(months))
    yr = sub.time.dt.year.values.astype(int)
    if season == "Dec–Feb":
        yr = yr + (sub.time.dt.month.values == 12).astype(int)
    sub = sub.assign_coords(year=("time", yr))
    counts = pd.Series(yr).value_counts()
    ok = sorted(counts[counts == len(months)].index)
    return sub.groupby("year").mean("time", skipna=True).sel(year=ok).load()

@st.cache_data(show_spinner="Computing trend at every grid cell…")
def grid_trend(name, season, y0, y1):
    a = load_seasonal(name, season).sel(year=slice(y0, y1))
    A = a.values
    X = a["year"].values.astype(float)[:, None, None] * np.ones_like(A)
    m = ~np.isnan(A)
    n = m.sum(0).astype(float)
    with np.errstate(all="ignore"):
        xm = np.where(m, X, 0).sum(0) / n
        ym = np.where(m, A, 0).sum(0) / n
        dx = np.where(m, X - xm, 0); dy = np.where(m, A - ym, 0)
        sxx, sxy, syy = (dx**2).sum(0), (dx*dy).sum(0), (dy**2).sum(0)
        slope = sxy / sxx
        se = np.sqrt((syy - slope*sxy) / (n - 2) / sxx)
        p = 2 * stats.t.sf(np.abs(slope / se), n - 2)
    bad = n < 10
    slope[bad] = np.nan; p[bad] = np.nan
    return (xr.DataArray(slope, coords=[a.lat, a.lon]),
            xr.DataArray(p, coords=[a.lat, a.lon]))

def fdr_mask(p_da, alpha):
    """Benjamini-Hochberg false discovery rate: boolean mask of cells that stay significant."""
    pv = p_da.values
    flat = pv.ravel()
    ok = np.isfinite(flat)
    pvals = flat[ok]
    m = pvals.size
    out = np.zeros(flat.shape, dtype=bool)
    if m == 0:
        return out.reshape(pv.shape)
    ranked = np.sort(pvals)
    passed = np.where(ranked <= alpha * np.arange(1, m + 1) / m)[0]
    if passed.size:
        out[ok] = pvals <= ranked[passed.max()]
    return out.reshape(pv.shape)

@st.cache_data
def coastlines():
    if not os.path.exists("data/coast.geojson"): return [], []
    xs, ys = [], []
    for f in json.load(open("data/coast.geojson"))["features"]:
        g = f["geometry"]
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for ln in lines:
            xs += [p[0] for p in ln] + [None]; ys += [p[1] for p in ln] + [None]
    return xs, ys

def regional_series(name, season, y0, y1, box):
    lat0, lat1, lon0, lon1 = box
    reg = load_seasonal(name, season).sel(year=slice(y0, y1),
                                          lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    w = np.cos(np.deg2rad(reg.lat))
    return reg.weighted(w).mean(("lat", "lon")).dropna("year")

def globe_scale(cmap):
    """Diverging scales whose middle is dark, so 'no change' fades into space."""
    mid = "#14224a"
    if cmap == "BrBG":      # low = drier/browning, high = wetter/greening
        return [[0, "#e0952b"], [0.25, "#8a5a1c"], [0.5, mid], [0.75, "#1f9e7a"], [1, "#3dffb0"]]
    if cmap == "RdBu":      # low = loss (red), high = gain (blue)
        return [[0, "#ff2d2d"], [0.25, "#ff8a65"], [0.5, mid], [0.75, "#1e6fd9"], [1, "#2ec5ff"]]
    return [[0, "#2ec5ff"], [0.25, "#1e6fd9"], [0.5, mid], [0.75, "#ff8a65"], [1, "#ff2d2d"]]

def sph(lat, lon, r=1.0):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    return r * np.cos(phi) * np.cos(lam), r * np.cos(phi) * np.sin(lam), r * np.sin(phi)

def build_globe(slope_dec, sig_map, cfg, box, dim, title):
    lat = slope_dec.lat.values
    lon = slope_dec.lon.values
    z = slope_dec.values.astype(float)
    sig = np.asarray(sig_map, dtype=bool)

    vmax = float(np.nanpercentile(np.abs(z), 98)) if np.isfinite(z).any() else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    shown = np.clip(z, -vmax, vmax)
    keep = sig if dim else np.ones_like(sig)
    shown = np.nan_to_num(np.where(keep, shown, shown * 0.3), nan=0.0)

    # close the poles and the seam at 180 degrees so there are no gaps
    lat_e = np.concatenate([[-90.0], lat, [90.0]])
    lon_e = np.concatenate([lon, [lon[0] + 360.0]])
    def ext(a):
        a = np.vstack([a[:1], a, a[-1:]])
        return np.hstack([a, a[:, :1]])
    C, Zt, S = ext(shown), ext(z), ext(sig)
    LON, LAT = np.meshgrid(lon_e, lat_e)
    X, Y, Z = sph(LAT, LON)

    txt = [[("no data" if not np.isfinite(Zt[i, j]) else
             f"lat {LAT[i, j]:.0f}°, lon {((LON[i, j] + 180) % 360) - 180:.0f}°<br>"
             f"trend {Zt[i, j]:+.3f} {cfg['units']}/decade<br>"
             + ("significant" if S[i, j] else "not significant"))
            for j in range(C.shape[1])] for i in range(C.shape[0])]

    surf = go.Surface(
        x=X, y=Y, z=Z, surfacecolor=C, cmin=-vmax, cmax=vmax, colorscale=globe_scale(cfg["cmap"]),
        text=txt, hoverinfo="text", showscale=True,
        lighting=dict(ambient=0.95, diffuse=0.25, specular=0.08, roughness=0.9),
        colorbar=dict(title=f"{cfg['units']}<br>per decade", thickness=14, len=0.6),
        contours=dict(x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False)))
    traces = [surf]

    cx, cy = coastlines()
    if cx:
        ax = np.array([np.nan if v is None else v for v in cx], dtype=float)
        ay = np.array([np.nan if v is None else v for v in cy], dtype=float)
        gx, gy, gz = sph(ay, ax, 1.004)
        traces.append(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines",
                                   line=dict(color="rgba(255,255,255,0.55)", width=1.5),
                                   hoverinfo="skip", showlegend=False))

    lat0, lat1, lon0, lon1 = box
    is_global = (lat1 - lat0) >= 180 and (lon1 - lon0) >= 360
    if not is_global:
        n = 60
        lo, la = np.linspace(lon0, lon1, n), np.linspace(lat0, lat1, n)
        path_lat = np.concatenate([np.full(n, lat0), la, np.full(n, lat1), la[::-1]])
        path_lon = np.concatenate([lo, np.full(n, lon1), lo[::-1], np.full(n, lon0)])
        bx, by, bz = sph(path_lat, path_lon, 1.008)
        traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color="gold", width=6),
                                   hoverinfo="skip", showlegend=False))

    clat, clon = (20.0, 20.0) if is_global else ((lat0 + lat1) / 2, (lon0 + lon1) / 2)
    ex, ey, ez = sph(clat, clon, 1.7)

    frames = []
    for k in range(0, 360, 3):
        fx, fy, fz = sph(clat, clon + k, 1.7)
        frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=fx, y=fy, z=fz)))), name=str(k)))

    spin_buttons = dict(
        type="buttons", direction="left", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
        showactive=False, bgcolor="rgba(14,22,60,.85)", bordercolor="rgba(79,195,247,.5)",
        font=dict(color="#dbe7ff"),
        buttons=[
            dict(label="▶ Spin", method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                    mode="immediate")]),
        ])

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        height=620, title=title, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe7ff", family="Rajdhani, sans-serif", size=14),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=ex, y=ey, z=ez), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))),
        updatemenus=[spin_buttons])
    return fig

def boxes_overlap(b1, b2):
    """Do two (lat0, lat1, lon0, lon1) boxes overlap?"""
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])

def run_scan(season, y0, y1, test_name, alpha, use_fdr, progress=None):
    """Test every variable in every preset region; return one row per combination."""
    regs = [(rn, bx) for rn, bx in REGIONS.items() if bx is not None]
    total = max(len(VARIABLES) * len(regs), 1)
    rows, i = [], 0
    for nm, c in VARIABLES.items():
        for rn, bx in regs:
            i += 1
            if progress is not None:
                progress.progress(i / total, text=f"Scanning… {i}/{total}")
            try:
                s = regional_series(nm, season, y0, y1, bx)
                if len(s) < 8:
                    continue
                r = TESTS[test_name](s.values, alpha=alpha)
                sd = float(s.std())
                if not np.isfinite(sd) or sd == 0:
                    continue
                rows.append(dict(Variable=nm, Region=rn, Trend=float(r.slope * 10), Units=c["units"],
                                 Strength=float(r.slope * 10 / sd), p=float(r.p),
                                 Significant=bool(r.h), Years=len(s), Box=bx))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if len(df) and use_fdr:
        df["Significant"] = fdr_mask(df["p"], alpha)
    return df

def jump_to(var_kw, region_kw, cmp_kw, s, a, b):
    """Callback: jump the controls to a finding from the Discover tab."""
    st.session_state["defaults"] = dict(var_kw=var_kw, season=s, region_kw=region_kw, cmp_kw=cmp_kw, y0=a, y1=b)
    st.session_state["ver"] = st.session_state.get("ver", 0) + 1

CASES = {
    "🧊 Arctic sea ice":     dict(var_kw="Sea ice", season="Sep–Nov", region_kw="Arctic", cmp_kw="Antarctic", y0=1979),
    "🌿 Sahel greening":     dict(var_kw="NDVI", season="Annual", region_kw="Sahel", cmp_kw="Amazon", y0=2001),
    "🌧️ Indonesia rainfall": dict(var_kw="Precipitation", season="Annual", region_kw="Indonesia", cmp_kw="Sahel", y0=1980),
    "🔥 Warming Arctic":     dict(var_kw="Surface temperature", season="Annual", region_kw="Arctic", cmp_kw="Antarctic", y0=1980),
}
version = st.session_state.get("ver", 0)
preset = st.session_state.get("defaults", {})

def index_of(options, kw, default=0):
    return next((i for i, o in enumerate(options) if kw and kw in o), default)

if not REC:
    st.markdown("##### 🎯 Quick case studies")
    case_cols = st.columns(len(CASES))
    for col, (label, settings) in zip(case_cols, CASES.items()):
        if col.button(label, use_container_width=True, key=f"case_{label}"):
            st.session_state["defaults"] = settings
            st.session_state["ver"] = version + 1
            st.rerun()
    st.write("")

st.sidebar.markdown("## ⚙️ Controls")
variable_names = list(VARIABLES)
var_name = st.sidebar.selectbox("Variable", variable_names, index=index_of(variable_names, preset.get("var_kw")), key=f"var_{version}")
cfg = VARIABLES[var_name]
annual = load_seasonal(var_name, "Annual")
ymin, ymax = int(annual.year.min()), int(annual.year.max())

season = st.sidebar.radio("Season", list(SEASONS), index=list(SEASONS).index(preset.get("season", "Annual")),
                          horizontal=True, key=f"season_{version}")
start_default = min(max(ymin, preset.get("y0", 1980)), ymax - 1)
end_default = max(min(preset.get("y1", ymax), ymax), start_default + 1)
y0, y1 = st.sidebar.slider("Time range", ymin, ymax, (start_default, end_default), key=f"yrs_{version}_{var_name}")

region_names = list(REGIONS)
region_name = st.sidebar.selectbox("Region", region_names, index=index_of(region_names, preset.get("region_kw")), key=f"region_{version}")
if REGIONS[region_name] is None:
    lat0, lat1 = st.sidebar.slider("Latitude", -90, 90, (-30, 30))
    lon0, lon1 = st.sidebar.slider("Longitude", -180, 180, (-180, 180))
    box = (lat0, lat1, lon0, lon1)
else:
    box = REGIONS[region_name]
    st.sidebar.caption(f"Box: {box[0]}° to {box[1]}° lat, {box[2]}° to {box[3]}° lon")

alpha = st.sidebar.select_slider("Significance level (α)", [0.01, 0.05, 0.1], value=0.05)
test_name = st.sidebar.selectbox("Statistical test", list(TESTS))
use_fdr = st.sidebar.checkbox("Correct map for multiple testing (FDR)", value=True,
                              help="Thousands of grid cells are tested at once, so some look significant by chance. "
                                   "The Benjamini-Hochberg false discovery rate correction removes many of those false positives.")
st.sidebar.checkbox("🎬 Recording layout", key="rec_mode",
                    help="Hides the banner, case-study buttons and top result cards, so the globe and its "
                         "result cards fit on one screen without scrolling.")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Source:** {cfg['source']}  \n{cfg['info']}")

slope, p = grid_trend(var_name, season, y0, y1)
slope_dec = slope * 10
p_values = p.values
sig_map = fdr_mask(p, alpha) if use_fdr else (p_values < alpha)
sig_da = xr.DataArray(sig_map, coords=[p.lat, p.lon])
n_valid = max(int(np.isfinite(p_values).sum()), 1)
raw_pct = float((p_values < alpha).sum()) / n_valid * 100
adj_pct = float(sig_map.sum()) / n_valid * 100
series = regional_series(var_name, season, y0, y1, box)
if len(series) < 8:
    st.warning("Not enough data in this region / time range. Try a wider range."); st.stop()

res = TESTS[test_name](series.values, alpha=alpha)
years = series.year.values
trend_line = res.intercept + res.slope * np.arange(len(years))
per_dec = res.slope * 10

lat0, lat1, lon0, lon1 = box
rs = slope.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
rsig = sig_da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
valid = max(int(rs.notnull().sum()), 1)
pos = float(((rs > 0) & rsig).sum()) / valid * 100
neg = float(((rs < 0) & rsig).sum()) / valid * 100

up = res.trend == "increasing"; dn = res.trend == "decreasing"
accent = "#ff6e6e" if up else "#4fc3f7" if dn else "#8892a0"
arrow = "▲" if up else "▼" if dn else "■"
kpi_cards = [
    kpi("What", plain(var_name), f"{season} · {y0}–{y1}"),
    kpi("How much", f"{arrow} {per_dec:+.3f}", f"{cfg['units']} per decade (Sen's slope)", accent),
    kpi("p-value", f"{res.p:.4f}", f"α = {alpha}"),
    kpi("Significant?", "Yes ✅" if res.h else "No ❌",
        "Trend is unlikely due to chance" if res.h else "Cannot rule out chance",
        "#7dffb2" if res.h else "#8892a0"),
]
if not REC:
    for column, card in zip(st.columns(4), kpi_cards):
        column.markdown(card, unsafe_allow_html=True)
    st.write("")

tab_globe, tab_map, tab_ts, tab_cmp, tab_disc, tab_rank, tab_about = st.tabs(
    ["🌍 Globe", "🗺️ Flat map", "📈 Time series", "⚖️ Compare regions", "🔎 Discover",
     "🏆 Variable ranking", "ℹ️ Method & caveats"])

with tab_globe:
    globe_col, cards_col = st.columns([3.1, 1])
    with cards_col:
        for card in kpi_cards:
            st.markdown(card, unsafe_allow_html=True)
        dim = st.checkbox("Dim non-significant areas", value=True, key="globe_dim")
        st.caption("Bright = significant, muted = not. Gold outline = your region.")
    with globe_col:
        with st.spinner("Rendering globe…"):
            globe = build_globe(slope_dec, sig_map, cfg, box, dim,
                                f"{plain(var_name)} · {season} · trend {y0}–{y1}")
        st.plotly_chart(globe, use_container_width=True, theme=None)

with tab_map:
    vmax = float(np.nanpercentile(np.abs(slope_dec.values), 98))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=slope_dec.lon, y=slope_dec.lat, z=slope_dec.values,
        colorscale=cfg["cmap"], zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title=f"{cfg['units']}<br>per decade", thickness=14),
        hovertemplate="lon %{x}°<br>lat %{y}°<br>trend %{z:.3f}<extra></extra>"))
    yy, xx = np.where(sig_map)
    fig.add_trace(go.Scattergl(x=slope_dec.lon.values[xx], y=slope_dec.lat.values[yy],
                               mode="markers", marker=dict(size=2, color="rgba(255,255,255,.7)"),
                               name=f"Significant (p<{alpha})", hoverinfo="skip"))
    cx, cy = coastlines()
    fig.add_trace(go.Scattergl(x=cx, y=cy, mode="lines", line=dict(color="rgba(255,255,255,.65)", width=.8),
                               hoverinfo="skip", name="Coastlines", showlegend=False))
    fig.add_shape(type="rect", x0=lon0, x1=lon1, y0=lat0, y1=lat1,
                  line=dict(color="gold", width=3))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"Linear trend, {season}, {y0}–{y1}  (dots = significant{', FDR-corrected' if use_fdr else ''})",
                      xaxis=dict(range=[-180, 180], title="Longitude", showgrid=False),
                      yaxis=dict(range=[-90, 90], title="Latitude", showgrid=False),
                      legend=dict(orientation="h", y=-0.12))
    show(fig)
    st.caption(f"Cells that look significant at α = {alpha}: {raw_pct:.0f}% uncorrected"
               + (f" → {adj_pct:.0f}% after false-discovery-rate correction." if use_fdr else "."))

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Opposite trends in the same region?")
        st.markdown(
            f'<span class="pill up">▲ {pos:.0f}% of cells significantly increasing</span><br><br>'
            f'<span class="pill down">▼ {neg:.0f}% of cells significantly decreasing</span><br><br>'
            f'<span class="pill no">{100-pos-neg:.0f}% no significant trend</span>',
            unsafe_allow_html=True)
    with right:
        prof = rs.mean("lon")
        bars = go.Figure(go.Bar(
            y=prof.lat.values, x=(prof * 10).values, orientation="h",
            marker_color=np.where(prof.values >= 0, "#ff6e6e", "#4fc3f7")))
        bars.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                           title="Trend by latitude (averaged across the box)",
                           xaxis_title=f"{cfg['units']} per decade", yaxis_title="Latitude")
        show(bars)

with tab_ts:
    ts = go.Figure()
    ts.add_trace(go.Scatter(x=years, y=series.values, mode="lines+markers", name="Regional mean",
                            line=dict(color="#4fc3f7", width=2), marker=dict(size=5)))
    ts.add_trace(go.Scatter(x=years, y=trend_line, name="Sen's slope trend",
                            line=dict(color="#ff6e6e", width=3, dash="dash")))
    ts.update_layout(height=420, title="Area-weighted regional time series",
                     xaxis_title="Year", yaxis_title=cfg["units"], margin=dict(t=50))
    show(ts)

    direction = {"increasing": "increased", "decreasing": "decreased",
                 "no trend": "shown no clear trend"}[res.trend]
    st.markdown(
        f'<div class="callout" style="border-color:{accent}"><b>Plain-language summary</b><br>'
        f'In <b>{plain(region_name)}</b>, <b>{plain(var_name).lower()}</b> '
        f'({season.lower()}) has <b>{direction}</b>'
        + (f' by about <b>{abs(per_dec):.2f} {cfg["units"]} per decade</b>' if res.trend != "no trend" else "")
        + f' between <b>{y0}</b> and <b>{y1}</b>. This is '
        f'<b>{"statistically significant" if res.h else "not statistically significant"}</b> '
        f'(p = {res.p:.4f}, α = {alpha}, {test_name.split(" (")[0]}). '
        f'Inside the region, {pos:.0f}% of grid cells are significantly rising and '
        f'{neg:.0f}% significantly falling.</div>', unsafe_allow_html=True)
    st.write("")
    df = pd.DataFrame({"year": years, "value": series.values, "trend": trend_line})
    st.download_button("⬇️ Download this series (CSV)", df.to_csv(index=False),
                       file_name="regional_series.csv", mime="text/csv")

with tab_cmp:
    st.markdown("Pick a second region. If the same variable trends in **opposite directions** "
                "in the two places, the tool flags it.")
    other_regions = [r for r in REGIONS if REGIONS[r] is not None and r != region_name]
    region_b = st.selectbox("Compare with", other_regions, index=index_of(other_regions, preset.get("cmp_kw")), key=f"cmp_{version}")
    s_b = regional_series(var_name, season, y0, y1, REGIONS[region_b])
    if len(s_b) < 8:
        st.warning("Not enough data in the second region for this period.")
    else:
        res_b = TESTS[test_name](s_b.values, alpha=alpha)
        years_b = s_b.year.values
        line_b = res_b.intercept + res_b.slope * np.arange(len(years_b))
        name_a = plain(region_name)
        name_b = plain(region_b)

        def color_for(r):
            return "#ff6e6e" if r.trend == "increasing" else "#4fc3f7" if r.trend == "decreasing" else "#8892a0"
        def arrow_for(r):
            return "▲" if r.trend == "increasing" else "▼" if r.trend == "decreasing" else "■"
        def card_note(r):
            return f"{cfg['units']}/decade · p = {r.p:.4f} · " + ("significant" if r.h else "not significant")

        ca, cb = st.columns(2)
        ca.markdown(kpi(name_a, f"{arrow_for(res)} {per_dec:+.3f}", card_note(res), color_for(res)), unsafe_allow_html=True)
        cb.markdown(kpi(name_b, f"{arrow_for(res_b)} {res_b.slope*10:+.3f}", card_note(res_b), color_for(res_b)), unsafe_allow_html=True)
        st.write("")

        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Scatter(x=years, y=series.values, mode="lines+markers", name=name_a,
                                     line=dict(color="#4fc3f7", width=2), marker=dict(size=5)))
        cmp_fig.add_trace(go.Scatter(x=years, y=trend_line, name=f"{name_a} trend",
                                     line=dict(color="#4fc3f7", width=3, dash="dash")))
        cmp_fig.add_trace(go.Scatter(x=years_b, y=s_b.values, mode="lines+markers", name=name_b,
                                     line=dict(color="#ffab40", width=2), marker=dict(size=5)))
        cmp_fig.add_trace(go.Scatter(x=years_b, y=line_b, name=f"{name_b} trend",
                                     line=dict(color="#ffab40", width=3, dash="dash")))
        cmp_fig.update_layout(height=420, title=f"{plain(var_name)}: {name_a} vs {name_b}",
                              xaxis_title="Year", yaxis_title=cfg["units"], margin=dict(t=50))
        show(cmp_fig)

        if res.h and res_b.h and np.sign(res.slope) != np.sign(res_b.slope):
            verdict, vcol = "🔀 <b>Opposite trends:</b> both regions show a statistically significant trend, but in opposite directions.", "#ffd54f"
        elif res.h and res_b.h:
            verdict, vcol = "Both regions show a significant trend in the <b>same direction</b>.", "#4fc3f7"
        elif res.h or res_b.h:
            verdict, vcol = "Only <b>one</b> of the two regions shows a statistically significant trend.", "#4fc3f7"
        else:
            verdict, vcol = "<b>Neither</b> region shows a statistically significant trend.", "#8892a0"
        st.markdown(f'<div class="callout" style="border-color:{vcol}"><b>Verdict</b><br>{verdict}</div>',
                    unsafe_allow_html=True)

with tab_disc:
    st.markdown("Not sure where to look? Scan **every variable in every preset region** using your current "
                "season, time range, test and significance settings, and let the tool surface the most "
                "interesting changes.")
    scan_settings = (season, y0, y1, test_name, alpha, use_fdr)
    if st.button("🔎 Scan all variables and regions", key="scan_btn"):
        progress_bar = st.progress(0.0, text="Scanning…")
        st.session_state["disc"] = dict(params=scan_settings, df=run_scan(*scan_settings, progress=progress_bar))
        progress_bar.empty()

    scan = st.session_state.get("disc")
    if scan is None:
        st.info("Press the button to start. A full scan takes a few seconds.")
    elif scan["df"].empty:
        st.warning("No variable–region combination had enough data for these settings.")
    else:
        if scan["params"] != scan_settings:
            st.warning("These results come from different settings (season, years, test, α or FDR). "
                       "Press the button to refresh them.")
        scan_season, scan_start, scan_end, scan_test, scan_alpha, scan_fdr = scan["params"]
        df = scan["df"]
        sig_df = df[df["Significant"]]
        st.markdown(f"**{len(sig_df)} of {len(df)}** variable–region combinations show a statistically "
                    f"significant trend ({scan_season.lower()}, {scan_start}–{scan_end}).")
        st.caption(("Significance controls the false discovery rate across all tests."
                    if scan_fdr else
                    f"No multiple-testing correction is on, so around {scan_alpha*len(df):.0f} of these "
                    "could be chance. Treat them as leads to investigate."))

        regs = [r for r in REGIONS if REGIONS[r] is not None]
        vars_ = list(VARIABLES)
        zmat = np.full((len(vars_), len(regs)), np.nan)
        tmat = [[""] * len(regs) for _ in vars_]
        hmat = [[""] * len(regs) for _ in vars_]
        for r in df.itertuples():
            i, j = vars_.index(r.Variable), regs.index(r.Region)
            zmat[i, j] = r.Strength
            tmat[i][j] = "✔" if r.Significant else ""
            hmat[i][j] = (f"{r.Variable} · {r.Region}<br>{r.Trend:+.3f} {r.Units}/decade<br>"
                          f"p = {r.p:.4f}" + (" · significant" if r.Significant else " · not significant"))
        zm = float(np.nanpercentile(np.abs(zmat), 95)) if np.isfinite(zmat).any() else 1.0
        zm = zm if zm > 0 else 1.0
        heat = go.Figure(go.Heatmap(
            z=zmat, x=[plain(r).split(" (")[0] for r in regs], y=vars_,
            text=tmat, texttemplate="%{text}", textfont=dict(color="white", size=16),
            hovertext=hmat, hoverinfo="text", zmid=0, zmin=-zm, zmax=zm,
            colorscale=[[0, "#2ec5ff"], [0.25, "#1e6fd9"], [0.5, "#14224a"], [0.75, "#ff8a65"], [1, "#ff2d2d"]],
            colorbar=dict(title="σ per decade<br>(red ↑ blue ↓)", thickness=14)))
        heat.update_layout(height=130 + 52 * len(vars_), title="Everything at a glance (✔ = significant)",
                           yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-35),
                           margin=dict(l=10, r=10, t=50, b=10))
        show(heat)

        st.markdown("#### 🏆 Strongest significant changes")
        top = sig_df.assign(abs_strength=sig_df["Strength"].abs()).sort_values("abs_strength", ascending=False).head(5)
        if top.empty:
            st.info("Nothing significant under these settings.")
        for k, r in enumerate(top.itertuples()):
            c1, c2 = st.columns([5, 1.3])
            arrow = "▲" if r.Trend > 0 else "▼"
            col = "#ff6e6e" if r.Trend > 0 else "#4fc3f7"
            c1.markdown(f'<div class="callout" style="border-color:{col}"><b>{plain(r.Variable)}</b> in '
                        f'<b>{plain(r.Region)}</b>: {arrow} {r.Trend:+.3f} {r.Units} per decade '
                        f'(p = {r.p:.4f})</div>', unsafe_allow_html=True)
            c2.button("Show it", key=f"top_{k}", use_container_width=True, on_click=jump_to,
                      args=(r.Variable, r.Region, None, scan_season, scan_start, scan_end))

        st.markdown("#### 🔀 Opposite trends in different places")
        st.caption("Same variable, two non-overlapping regions, both significant, opposite directions.")
        global_name = next((r for r in REGIONS if "Global" in r), None)
        pairs = []
        for vn, g in sig_df.groupby("Variable"):
            rows = list(g[g["Region"] != global_name].itertuples())
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    ra, rb = rows[a], rows[b]
                    if np.sign(ra.Trend) == np.sign(rb.Trend) or boxes_overlap(ra.Box, rb.Box):
                        continue
                    pairs.append((min(abs(ra.Strength), abs(rb.Strength)), vn, ra, rb))
        pairs.sort(key=lambda t: -t[0])
        if not pairs:
            st.info("No pair of non-overlapping regions shows significant opposite trends under these settings.")
        for k, (sc, vn, ra, rb) in enumerate(pairs[:5]):
            c1, c2 = st.columns([5, 1.3])
            c1.markdown(
                f'<div class="callout" style="border-color:#ffd54f"><b>{plain(vn)}</b>: '
                f'<b>{plain(ra.Region)}</b> {"▲" if ra.Trend > 0 else "▼"} {ra.Trend:+.3f} '
                f'vs <b>{plain(rb.Region)}</b> {"▲" if rb.Trend > 0 else "▼"} {rb.Trend:+.3f} '
                f'{ra.Units} per decade</div>', unsafe_allow_html=True)
            c2.button("Compare", key=f"pair_{k}", use_container_width=True, on_click=jump_to,
                      args=(vn, ra.Region, rb.Region, scan_season, scan_start, scan_end))
        st.caption("After pressing a button, open the Globe or Compare regions tab to see it.")

        with st.expander("Full results table"):
            table = df.drop(columns=["Box"]).sort_values("Strength", key=np.abs, ascending=False)
            st.dataframe(table.style.format({"Trend": "{:+.3f}", "Strength": "{:+.2f}", "p": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download scan (CSV)", table.to_csv(index=False),
                               file_name="scan_results.csv", mime="text/csv")

with tab_rank:
    st.markdown("Which variable is changing the most **in the selected region and period**? "
                "Trends are standardized (trend per decade ÷ year-to-year variability) so "
                "different units can be compared.")
    rows = []
    with st.spinner("Testing every variable…"):
        for nm, c in VARIABLES.items():
            try:
                s = regional_series(nm, season, y0, y1, box)
                if len(s) < 8: continue
                r = TESTS[test_name](s.values, alpha=alpha)
                rows.append(dict(Variable=nm, Trend_per_decade=r.slope * 10, Units=c["units"],
                                 Strength=r.slope * 10 / float(s.std()), p_value=r.p,
                                 Significant=bool(r.h), Years=len(s)))
            except Exception:
                pass
    if rows:
        rk = pd.DataFrame(rows).sort_values("Strength", key=np.abs, ascending=False)
        bar = go.Figure(go.Bar(
            y=rk["Variable"], x=rk["Strength"], orientation="h",
            marker_color=["#2ecc71" if s else "#6b7585" for s in rk["Significant"]],
            text=[("✔ significant" if s else "not significant") for s in rk["Significant"]]))
        bar.update_layout(height=120 + 55*len(rk),
                          xaxis_title="Standardized trend (σ per decade)",
                          yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20))
        show(bar)
        st.dataframe(rk.style.format({"Trend_per_decade": "{:+.3f}", "Strength": "{:+.2f}",
                                      "p_value": "{:.4f}"}), use_container_width=True, hide_index=True)
    else:
        st.info("No variables have enough data for this region and period.")

with tab_about:
    st.markdown(f"""
**How it works**
1. Monthly gridded data is averaged into {season.lower()} values for each year (only complete years/seasons are kept).
2. **Map:** a linear regression at every grid cell gives the trend and a p-value. Dots mark p < α.
3. **Region:** values inside the box are area-weighted (cos latitude) into one time series.
4. **Globe and Discover:** the globe shows the same per-cell trends on a sphere. Discover tests every variable in every preset region and applies a false discovery rate correction across all of those tests.
5. **Test:** Mann-Kendall for the trend's significance and Sen's slope for its size. Both are standard in climate science and robust to outliers.
   The Hamed-Rao and Yue-Wang options correct for autocorrelation, which otherwise makes trends look more significant than they are.

**Caveats**
- Short periods are strongly affected by natural variability (El Niño, volcanic eruptions).
- Satellite records can contain sensor changes; reanalysis variables are model output and not direct observations.
- The map uses ordinary regression for speed; the regional test uses Mann-Kendall.
- Thousands of map cells are tested at once, so some look significant by chance. The map dots use a Benjamini-Hochberg false discovery rate correction (toggle in the sidebar); the caption under the map shows how many cells survive.
- Neighbouring cells are correlated, so the FDR correction is approximate, not exact.
""")
    st.caption("Data: NASA GISS (GISTEMP), NASA MODIS (NDVI), NOAA PSL / NCEI, GPCP, NCEP/NCAR reanalysis, JMA COBE-SST2, Natural Earth coastlines. Built for NASA Space Apps. Code written with AI assistance (Claude, Anthropic) and reviewed by the team.")
