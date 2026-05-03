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
st.caption(f"Live data from MLB Stats API · Refreshed {datetime.now().strftime('%b %d, %Y %I:%M %p')}")

# ── Park factors ───────────────────────────────────────────────────────────────
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

# ── Fetch data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stats(season: int, min_pa: int) -> pd.DataFrame:
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&gameType=R"
        f"&season={season}&limit=300&sortStat=homeRuns"
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
        stat   = s.get("stat", {})
        player = s.get("player", {})
        team   = s.get("team", {})

        pa = int(stat.get("plateAppearances", 0) or 0)
        if pa < min_pa:
            continue

        hr      = int(stat.get("homeRuns", 0) or 0)
        ab      = int(stat.get("atBats", 1) or 1)
        rbi     = int(stat.get("rbi", 0) or 0)
        bb      = int(stat.get("baseOnBalls", 0) or 0)
        so      = int(stat.get("strikeOuts", 0) or 0)
        sb      = int(stat.get("stolenBases", 0) or 0)
        runs    = int(stat.get("runs", 0) or 0)
        doubles = int(stat.get("doubles", 0) or 0)
        triples = int(stat.get("triples", 0) or 0)

        def safe_float(key):
            v = stat.get(key)
            try:
                return float(v) if v not in (None, ".---", "-.--") else 0.0
            except (ValueError, TypeError):
                return 0.0

        avg = safe_float("avg")
        slg = safe_float("sluggingPercentage")
        obp = safe_float("obp")
        ops = safe_float("ops")
        iso = round(slg - avg, 3)

        hr_per_pa  = round(hr / pa, 4) if pa > 0 else 0.0
        ab_per_hr  = round(ab / hr, 1) if hr > 0 else 0.0
        k_pct      = round(so / pa * 100, 1) if pa > 0 else 0.0
        bb_pct     = round(bb / pa * 100, 1) if pa > 0 else 0.0

        team_name = team.get("name", "Unknown")
        pf        = PARK_FACTORS.get(team_name, 1.00)
        hr_score  = round(hr_per_pa * iso * pf * 10000, 1) if iso > 0 else 0.0

        rows.append({
            "Player":      player.get("fullName", "Unknown"),
            "Team":        team_name,
            "PA":          pa,
            "AB":          ab,
            "HR":          hr,
            "HR/PA":       hr_per_pa,
            "AB/HR":       ab_per_hr,
            "AVG":         avg,
            "OBP":         obp,
            "SLG":         slg,
            "OPS":         ops,
            "ISO":         iso,
            "RBI":         rbi,
            "Runs":        runs,
            "2B":          doubles,
            "3B":          triples,
            "BB":          bb,
            "BB%":         bb_pct,
            "K":           so,
            "K%":          k_pct,
            "SB":          sb,
            "Park Factor": pf,
            "HR Score":    hr_score,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("HR Score", ascending=False).reset_index(drop=True)
        df.index += 1
    return df


# ── Column groups for the sidebar ──────────────────────────────────────────────
COL_GROUPS = {
    "Home run stats":   ["HR", "HR/PA", "AB/HR", "HR Score"],
    "Power stats":      ["ISO", "SLG", "OPS"],
    "On-base stats":    ["AVG", "OBP", "BB", "BB%"],
    "Plate discipline": ["K", "K%", "PA", "AB"],
    "Run production":   ["RBI", "Runs", "2B", "3B", "SB"],
    "Ballpark":         ["Park Factor"],
}

DEFAULTS = {"HR", "HR/PA", "ISO", "HR Score", "Park Factor", "AVG", "OPS"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", [2026, 2025, 2024], index=0)
    min_pa = st.slider("Min plate appearances", 50, 300, 100, step=10)

    st.markdown("---")
    st.header("Columns to show")
    st.caption("Tick the stats you want in the table.")

    selected_cols = ["Player", "Team"]
    for group, cols in COL_GROUPS.items():
        st.markdown(f"**{group}**")
        for col in cols:
            if st.checkbox(col, value=(col in DEFAULTS), key=f"col_{col}"):
                selected_cols.append(col)

    st.markdown("---")
    st.markdown("**Sort table by**")
    sort_options = [c for c in selected_cols if c not in ("Player", "Team")]
    sort_by  = st.selectbox("", sort_options if sort_options else ["HR Score"],
                            label_visibility="collapsed")
    sort_asc = st.checkbox("Sort ascending", value=False)

    st.markdown("---")
    st.markdown("**HR Score formula**")
    st.caption("HR/PA × ISO × Park Factor × 10,000")
    st.caption("Higher = stronger HR prop candidate.")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Pulling live MLB data..."):
    df = fetch_stats(season, min_pa)

if df.empty:
    st.warning("No data returned. Try adjusting filters or check your connection.")
    st.stop()

# ── Player search ──────────────────────────────────────────────────────────────
search = st.text_input("Search for a player", placeholder="e.g. Aaron Judge")
if search:
    df = df[df["Player"].str.contains(search, case=False, na=False)]

# ── Sort ───────────────────────────────────────────────────────────────────────
if sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=sort_asc).reset_index(drop=True)
    df.index += 1

# ── Metric cards ───────────────────────────────────────────────────────────────
top = df.iloc[0] if not df.empty else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players shown", len(df))
if top is not None:
    c2.metric("HR leader",    top["Player"], f"{int(top['HR'])} HR")
    c3.metric("Best HR/PA",   top["Player"], f"{top['HR/PA']:.4f}")
    c4.metric("Top HR score", top["Player"], f"{top['HR Score']}")

st.markdown("---")

# ── Main table ─────────────────────────────────────────────────────────────────
st.subheader("Player rankings")

valid_cols = [c for c in selected_cols if c in df.columns]

format_map = {
    "HR/PA": "{:.4f}", "AVG": "{:.3f}", "OBP": "{:.3f}",
    "SLG": "{:.3f}",  "OPS": "{:.3f}", "ISO": "{:.3f}",
    "Park Factor": "{:.2f}", "BB%": "{:.1f}%", "K%": "{:.1f}%",
}
fmt = {k: v for k, v in format_map.items() if k in valid_cols}

styled = df[valid_cols].style.format(fmt)
if "HR Score"    in valid_cols: styled = styled.background_gradient(subset=["HR Score"],    cmap="YlOrRd")
if "HR/PA"       in valid_cols: styled = styled.background_gradient(subset=["HR/PA"],       cmap="Greens")
if "ISO"         in valid_cols: styled = styled.background_gradient(subset=["ISO"],         cmap="Purples")
if "Park Factor" in valid_cols: styled = styled.background_gradient(subset=["Park Factor"], cmap="Blues")
if "OPS"         in valid_cols: styled = styled.background_gradient(subset=["OPS"],         cmap="Oranges")

st.dataframe(styled, use_container_width=True, height=500)

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.subheader("Charts")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**HR/PA vs ISO — top 40**")
    top40 = df.head(40)
    if {"HR/PA", "ISO", "HR"}.issubset(top40.columns):
        fig = px.scatter(
            top40, x="ISO", y="HR/PA", text="Player",
            color="Park Factor", color_continuous_scale="RdYlGn",
            size="HR", hover_data=["Team", "HR", "HR Score"],
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Top 15 by HR Score**")
    top15 = df.head(15).sort_values("HR Score")
    if "HR Score" in top15.columns:
        fig2 = px.bar(
            top15, x="HR Score", y="Player", orientation="h",
            color="HR Score", color_continuous_scale="YlOrRd",
            hover_data=["HR", "HR/PA", "ISO", "Park Factor"],
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=380,
            yaxis_title="", coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API (statsapi.mlb.com) · "
    "Park factors are multi-year estimates · "
    "For informational purposes — bet responsibly."
)
