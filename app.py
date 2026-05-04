import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB HR Matchup Dashboard",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ MLB HR Matchup Dashboard")

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

# ── Helpers ────────────────────────────────────────────────────────────────────
def savant_name(row):
    """Build 'First Last' from Savant CSV which returns last_name, first_name."""
    if "last_name" in row.index and "first_name" in row.index:
        return f"{row['first_name'].strip()} {row['last_name'].strip()}"
    if "player_name" in row.index:
        name = str(row["player_name"])
        if "," in name:
            parts = [p.strip() for p in name.split(",")]
            return f"{parts[1]} {parts[0]}"
        return name
    return ""

def parse_savant_csv(text: str, col_map: dict) -> pd.DataFrame:
    """
    Read a Savant CSV, rename columns per col_map {savant_col: display_col},
    build Player name, return clean df.
    """
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()

    df["Player"] = df.apply(savant_name, axis=1)

    keep = {"Player"}
    rename = {}
    for src, dst in col_map.items():
        # fuzzy match — find first column containing src (case-insensitive)
        match = next((c for c in df.columns if src.lower() in c.lower()), None)
        if match:
            rename[match] = dst
            keep.add(match)

    df = df[list(keep & set(df.columns))].rename(columns=rename)
    for col in col_map.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Player"])


# ── Fetch schedule ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_schedule(game_date: str):
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}&hydrate=probablePitcher"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Could not load schedule: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD")
            home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD")
            away_team    = away.get("team", {}).get("name", "Unknown")
            home_team    = home.get("team", {}).get("name", "Unknown")
            away_id      = away.get("team", {}).get("id")
            home_id      = home.get("team", {}).get("id")
            game_time    = g.get("gameDate", "")
            venue        = g.get("venue", {}).get("name", "")
            try:
                dt       = datetime.strptime(game_time, "%Y-%m-%dT%H:%MZ")
                time_str = dt.strftime("%-I:%M %p") + " ET"
            except Exception:
                time_str = "TBD"
            games.append({
                "away_team":    away_team,  "home_team":    home_team,
                "away_id":      away_id,    "home_id":      home_id,
                "away_pitcher": away_pitcher, "home_pitcher": home_pitcher,
                "venue":        venue,      "time":         time_str,
                "label":        f"{away_team} @ {home_team}  —  {time_str}",
            })
    return games


# ── Fetch MLB API hitting stats ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_hitting_stats(season: int) -> pd.DataFrame:
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&gameType=R"
        f"&season={season}&limit=500&sortStat=homeRuns"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        splits = r.json()["stats"][0]["splits"]
    except Exception as e:
        st.error(f"Could not load hitting stats: {e}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat   = s.get("stat", {})
        player = s.get("player", {})
        team   = s.get("team", {})
        pa     = int(stat.get("plateAppearances", 0) or 0)
        if pa < 30:
            continue
        hr  = int(stat.get("homeRuns", 0) or 0)

        def sf(key):
            v = stat.get(key)
            try:
                return float(v) if v not in (None, ".---", "-.--") else 0.0
            except (ValueError, TypeError):
                return 0.0

        avg = sf("avg")
        slg = sf("sluggingPercentage")
        iso = round(slg - avg, 3)

        rows.append({
            "player_id": player.get("id"),
            "Player":    player.get("fullName", "Unknown"),
            "team_id":   team.get("id"),
            "Team":      team.get("name", "Unknown"),
            "PA": pa, "HR": hr, "ISO": iso, "SLG": slg,
        })
    return pd.DataFrame(rows)


# ── Fetch ALL Statcast stats in one call ───────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_statcast(season: int) -> pd.DataFrame:
    """
    Pull the Statcast batter summary from Baseball Savant.
    Returns one row per player with: HH%, Barrel%, EV, LA, FB%, Pull%
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&id=&sportId=1"
        f"&season={season}&season_end={season}"
        f"&min=50&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    df["Player"] = df.apply(savant_name, axis=1)

    # Map whatever Savant calls these columns → our display names
    col_map = {
        "hard_hit_percent":  "HH%",
        "barrel_batted_rate":"Barrel%",
        "launch_speed":      "Avg EV",
        "launch_angle":      "Avg LA",
        "fb_percent":        "FB%",
        "pull_percent":      "Pull%",
    }
    rename = {}
    for src, dst in col_map.items():
        match = next((c for c in df.columns if src.lower() in c.lower()), None)
        if match:
            rename[match] = dst

    df = df.rename(columns=rename)

    statcast_cols = ["Player"] + [c for c in col_map.values() if c in df.columns]
    df = df[statcast_cols].copy()
    for col in statcast_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Player"])


# ── Fetch pitcher stats ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_pitcher_stats(season: int) -> pd.DataFrame:
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&gameType=R"
        f"&season={season}&limit=500&sortStat=inningsPitched"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        splits = r.json()["stats"][0]["splits"]
    except Exception:
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat   = s.get("stat", {})
        player = s.get("player", {})
        try:
            ip = float(stat.get("inningsPitched", "0") or "0")
        except ValueError:
            ip = 0.0
        if ip < 10:
            continue
        hr_allowed = int(stat.get("homeRuns", 0) or 0)
        rows.append({
            "Pitcher": player.get("fullName", "Unknown"),
            "HR/9":    round(hr_allowed / ip * 9, 2) if ip > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ── Build matchup table ────────────────────────────────────────────────────────
STATCAST_COLS = ["HH%", "Barrel%", "Avg EV", "Avg LA", "FB%", "Pull%"]

def build_matchup_df(game, hitting_df, pitcher_df, min_pa, min_hr):
    rows = []
    matchups = [
        (game["away_id"], game["away_team"], game["home_pitcher"]),
        (game["home_id"], game["home_team"], game["away_pitcher"]),
    ]
    for batting_id, batting_team, opp_pitcher_name in matchups:
        batters = hitting_df[
            (hitting_df["team_id"] == batting_id) &
            (hitting_df["PA"] >= min_pa) &
            (hitting_df["HR"] >= min_hr)
        ]
        if batters.empty:
            continue

        park_factor  = PARK_FACTORS.get(game["home_team"], 1.00)
        pr           = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher_name]
        opp_hr9      = pr.iloc[0]["HR/9"] if not pr.empty else None
        pitcher_mult = 1.0 + (opp_hr9 - 1.2) * 0.15 if opp_hr9 is not None else 1.0

        for _, b in batters.iterrows():
            hh  = b.get("HH%",    40.0) or 40.0
            brl = b.get("Barrel%", 6.0) or 6.0
            score = round(
                b["ISO"] * (hh / 100) * (brl / 10)
                * park_factor * pitcher_mult * 100000, 1
            )
            row = {
                "Player":        b["Player"],
                "Batting team":  batting_team,
                "Opp pitcher":   opp_pitcher_name,
                "HR":            b["HR"],
                "ISO":           b["ISO"],
                "SLG":           b["SLG"],
                "Park factor":   park_factor,
                "Pitcher HR/9":  opp_hr9,
                "Matchup score": score,
            }
            # Add Statcast cols if present
            for col in STATCAST_COLS:
                row[col] = b.get(col, None)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Matchup score", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    season        = st.selectbox("Season", [2026, 2025, 2024], index=0)
    selected_date = st.date_input("Game date", value=date.today())
    min_pa        = st.slider("Min plate appearances", 30, 300, 80, step=10)
    min_hr        = st.slider("Min HR this season",     0,  20,  3, step=1)
    st.markdown("---")
    st.markdown("**Matchup score**")
    st.caption("ISO × HH% × Barrel% × Park factor × Pitcher multiplier")
    st.caption("Higher = stronger HR prop candidate today.")


# ── Load all data ──────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh hourly · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule, MLB stats, and Statcast data..."):
    games      = fetch_schedule(date_str)
    hitting_df = fetch_hitting_stats(season)
    statcast   = fetch_statcast(season)
    pitcher_df = fetch_pitcher_stats(season)

if not games:
    st.warning(f"No games found for {date_str}. Try a different date.")
    st.stop()
if hitting_df.empty:
    st.warning("Could not load hitting stats. Try again in a moment.")
    st.stop()

# ── Merge Statcast into hitting data ──────────────────────────────────────────
if not statcast.empty:
    hitting_df   = hitting_df.merge(statcast, on="Player", how="left")
    sc_loaded    = True
    sc_available = [c for c in STATCAST_COLS if c in hitting_df.columns]
else:
    for col in STATCAST_COLS:
        hitting_df[col] = None
    sc_loaded    = False
    sc_available = []

if not sc_loaded:
    st.warning(
        "⚠️ Baseball Savant data unavailable right now — Statcast columns will show '—'. "
        "All other data is current."
    )

# ── Game cards ─────────────────────────────────────────────────────────────────
st.subheader("Today's games")
gcols = st.columns(min(len(games), 3))
for i, g in enumerate(games):
    pf    = PARK_FACTORS.get(g["home_team"], 1.00)
    emoji = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    label = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral"
    with gcols[i % 3]:
        st.markdown(f"""
<div style="background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:12px;margin-bottom:10px">
  <div style="font-size:13px;font-weight:500;color:var(--color-text-primary)">{g['away_team']} @ {g['home_team']}</div>
  <div style="font-size:11px;color:var(--color-text-secondary);margin-top:2px">{g['time']} · {g['venue']}</div>
  <div style="font-size:11px;color:var(--color-text-tertiary);margin-top:6px">Away SP: {g['away_pitcher']}</div>
  <div style="font-size:11px;color:var(--color-text-tertiary)">Home SP: {g['home_pitcher']}</div>
  <div style="font-size:11px;margin-top:6px">{emoji} {label} · PF {pf:.2f}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Game selector ──────────────────────────────────────────────────────────────
st.subheader("HR prop rankings")
all_label = f"All games today ({len(games)} games)"
options   = [all_label] + [g["label"] for g in games]
choice    = st.selectbox("Filter by game", options)

# ── Build matchup data ─────────────────────────────────────────────────────────
if choice == all_label:
    frames = [build_matchup_df(g, hitting_df, pitcher_df, min_pa, min_hr) for g in games]
    frames = [f for f in frames if not f.empty]
    if frames:
        matchup_df = pd.concat(frames).sort_values("Matchup score", ascending=False).reset_index(drop=True)
        matchup_df.index += 1
    else:
        matchup_df = pd.DataFrame()
else:
    game       = next(g for g in games if g["label"] == choice)
    matchup_df = build_matchup_df(game, hitting_df, pitcher_df, min_pa, min_hr)

if matchup_df.empty:
    st.info("No batters match your filters. Try lowering Min PA or Min HR.")
    st.stop()

# ── Metric cards ───────────────────────────────────────────────────────────────
top = matchup_df.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Batters shown", len(matchup_df))
c2.metric("Top pick today", top["Player"], f"Score {top['Matchup score']}")
c3.metric("Facing", top["Opp pitcher"],
          f"HR/9: {top['Pitcher HR/9']:.2f}" if top["Pitcher HR/9"] is not None else "HR/9: TBD")
c4.metric("Park factor", top["Batting team"], f"PF {top['Park factor']:.2f}")

# ── Search ─────────────────────────────────────────────────────────────────────
search  = st.text_input("Search player or pitcher", placeholder="e.g. Judge, Cole")
view_df = matchup_df.copy()
if search:
    mask = (
        view_df["Player"].str.contains(search, case=False, na=False) |
        view_df["Opp pitcher"].str.contains(search, case=False, na=False)
    )
    view_df = view_df[mask]

# ── Table ──────────────────────────────────────────────────────────────────────
# Full column order — all always shown
display_cols = [
    "Player", "Batting team", "Opp pitcher",
    "HR",
    "HH%", "Barrel%", "Avg EV", "Avg LA", "FB%", "Pull%",  # Statcast
    "ISO", "SLG",                                            # MLB API
    "Park factor", "Pitcher HR/9",                           # Context
    "Matchup score",
]
display_cols = [c for c in display_cols if c in view_df.columns]

fmt = {
    "HH%":          "{:.1f}%",
    "Barrel%":      "{:.1f}%",
    "Avg EV":       "{:.1f}",
    "Avg LA":       "{:.1f}°",
    "FB%":          "{:.1f}%",
    "Pull%":        "{:.1f}%",
    "ISO":          "{:.3f}",
    "SLG":          "{:.3f}",
    "Park factor":  "{:.2f}",
    "Pitcher HR/9": "{:.2f}",
}

styled = view_df[display_cols].style.format(fmt, na_rep="—")
styled = styled.background_gradient(subset=["Matchup score"], cmap="YlOrRd")
styled = styled.background_gradient(subset=["ISO"],           cmap="Purples")
styled = styled.background_gradient(subset=["SLG"],           cmap="Blues")

# Only gradient Statcast cols if data loaded
for col, cmap in [("HH%","Greens"),("Barrel%","YlOrRd"),("Avg EV","Oranges"),("FB%","Blues"),("Pull%","Purples")]:
    if col in display_cols and view_df[col].notna().any():
        styled = styled.background_gradient(subset=[col], cmap=cmap)

if "Pitcher HR/9" in display_cols and view_df["Pitcher HR/9"].notna().any():
    styled = styled.background_gradient(subset=["Pitcher HR/9"], cmap="Reds")

st.dataframe(styled, use_container_width=True, height=520)

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.subheader("Charts")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Top 15 matchup scores**")
    top15 = matchup_df.head(15).sort_values("Matchup score")
    fig = px.bar(
        top15, x="Matchup score", y="Player", orientation="h",
        color="Matchup score", color_continuous_scale="YlOrRd",
        hover_data=["Batting team", "Opp pitcher", "HR", "HH%", "Barrel%"],
        text="Opp pitcher",
    )
    fig.update_traces(textposition="inside", textfont_size=9)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=420,
        yaxis_title="", coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Barrel% vs HH% — today's batters**")
    plot_df = matchup_df.head(60)
    if "Barrel%" in plot_df.columns and plot_df["Barrel%"].notna().any():
        fig2 = px.scatter(
            plot_df, x="HH%", y="Barrel%", text="Player",
            color="Matchup score", color_continuous_scale="YlOrRd",
            size="HR", hover_data=["Batting team", "Opp pitcher", "ISO", "Avg EV", "FB%"],
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Statcast data unavailable from Savant today — chart will appear once data loads.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API + Baseball Savant Statcast · "
    "Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
