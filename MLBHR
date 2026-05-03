import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB HR Betting Dashboard",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ MLB Home Run Betting Dashboard")
st.caption(f"Data pulled live from MLB Stats API · Updated {datetime.now().strftime('%b %d, %Y %I:%M %p')}")

# ── Park factors (2024/2025 estimates — update yearly) ─────────────────────────
PARK_FACTORS = {
    "Cincinnati Reds": 1.18, "Colorado Rockies": 1.15, "Philadelphia Phillies": 1.12,
    "Atlanta Braves": 1.10, "Texas Rangers": 1.09, "Chicago Cubs": 1.08,
    "Milwaukee Brewers": 1.07, "Boston Red Sox": 1.06, "Baltimore Orioles": 1.05,
    "Toronto Blue Jays": 1.04, "Arizona Diamondbacks": 1.03, "New York Yankees": 1.02,
    "Houston Astros": 1.01, "Cleveland Guardians": 1.00, "Minnesota Twins": 1.00,
    "Los Angeles Dodgers": 0.99, "San Diego Padres": 0.98, "Chicago White Sox": 0.98,
    "Kansas City Royals": 0.97, "Pittsburgh Pirates": 0.97, "Washington Nationals": 0.96,
    "New York Mets": 0.96, "Detroit Tigers": 0.95, "Seattle Mariners": 0.95,
    "Tampa Bay Rays": 0.94, "Los Angeles Angels": 0.94, "St. Louis Cardinals": 0.93,
    "San Francisco Giants": 0.92, "Miami Marlins": 0.91, "Oakland Athletics": 0.90,
}

# ── Fetch data from MLB Stats API ──────────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_hr_leaders(season: int, min_pa: int) -> pd.DataFrame:
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&gameType=R"
        f"&season={season}&limit=200&sortStat=homeRuns"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception as e:
        st.error(f"Could not reach MLB API: {e}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})
        pa = int(stat.get("plateAppearances", 0) or 0)
        hr = int(stat.get("homeRuns", 0) or 0)
        ab = int(stat.get("atBats", 1) or 1)
        hits = int(stat.get("hits", 0) or 0)
        slg_raw = stat.get("sluggingPercentage")
        avg_raw = stat.get("avg")

        if pa < min_pa:
            continue

        try:
            slg = float(slg_raw) if slg_raw not in (None, ".---") else 0.0
            avg = float(avg_raw) if avg_raw not in (None, ".---") else 0.0
        except (ValueError, TypeError):
            slg, avg = 0.0, 0.0

        iso = round(slg - avg, 3)
        hr_per_pa = round(hr / pa, 4) if pa > 0 else 0.0
        team_name = team.get("name", "Unknown")
        pf = PARK_FACTORS.get(team_name, 1.00)

        rows.append({
            "Player": player.get("fullName", "Unknown"),
            "Team": team_name,
            "PA": pa,
            "HR": hr,
            "HR/PA": hr_per_pa,
            "ISO": iso,
            "Park Factor": pf,
            "HR Score": round(hr_per_pa * iso * pf * 10000, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("HR Score", ascending=False).reset_index(drop=True)
        df.index += 1
    return df


# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", [2026, 2025, 2024], index=0)
    min_pa = st.slider("Minimum plate appearances", 50, 300, 100, step=10)
    st.markdown("---")
    st.markdown("**HR Score formula**")
    st.caption("HR/PA × ISO × Park Factor × 10,000")
    st.caption("Higher = better HR prop candidate today.")
    st.markdown("---")
    st.markdown("**Park factor guide**")
    st.caption("> 1.05 = hitter friendly")
    st.caption("1.00 = neutral")
    st.caption("< 0.95 = pitcher friendly")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Pulling live MLB data..."):
    df = fetch_hr_leaders(season, min_pa)

if df.empty:
    st.warning("No data returned. Try a different season or check your connection.")
    st.stop()

# ── Top metric cards ───────────────────────────────────────────────────────────
top = df.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Players loaded", len(df))
col2.metric("Top HR leader", f"{top['Player']}", f"{top['HR']} HR")
col3.metric("Best HR/PA", f"{top['Player']}", f"{top['HR/PA']:.4f}")
col4.metric("Top HR score today", top['Player'], f"{top['HR Score']}")

st.markdown("---")

# ── Main table ─────────────────────────────────────────────────────────────────
st.subheader("Player rankings")
st.caption("Sorted by HR Score — your best prop candidates at the top.")

show_cols = ["Player", "Team", "PA", "HR", "HR/PA", "ISO", "Park Factor", "HR Score"]

st.dataframe(
    df[show_cols].style
        .background_gradient(subset=["HR Score"], cmap="YlOrRd")
        .background_gradient(subset=["HR/PA"], cmap="Greens")
        .background_gradient(subset=["Park Factor"], cmap="Blues")
        .format({"HR/PA": "{:.4f}", "ISO": "{:.3f}", "Park Factor": "{:.2f}"}),
    use_container_width=True,
    height=480,
)

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("HR/PA vs ISO — top 40")
    top40 = df.head(40)
    fig = px.scatter(
        top40, x="ISO", y="HR/PA", text="Player",
        color="Park Factor", color_continuous_scale="RdYlGn",
        size="HR", hover_data=["Team", "HR", "HR Score"],
        title="",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Top 15 by HR Score")
    top15 = df.head(15).sort_values("HR Score")
    fig2 = px.bar(
        top15, x="HR Score", y="Player", orientation="h",
        color="HR Score", color_continuous_scale="YlOrRd",
        hover_data=["HR", "HR/PA", "ISO", "Park Factor"],
        title="",
    )
    fig2.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        yaxis_title="",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API (statsapi.mlb.com) · "
    "Park factors are estimates based on multi-year averages · "
    "This is for informational purposes — bet responsibly."
)
