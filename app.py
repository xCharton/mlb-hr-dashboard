import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import io
from sklearn.preprocessing import MinMaxScaler

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

# ── Savant name helper ─────────────────────────────────────────────────────────
def parse_savant_name(val: str) -> str:
    """'Doe, John' → 'John Doe'"""
    if "," in str(val):
        parts = [p.strip() for p in str(val).split(",")]
        return f"{parts[1]} {parts[0]}"
    return str(val).strip()

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
            game_time = g.get("gameDate", "")
            try:
                dt = datetime.strptime(game_time, "%Y-%m-%dT%H:%MZ")
                time_str = dt.strftime("%-I:%M %p") + " ET"
            except Exception:
                time_str = "TBD"
            games.append({
                "away_team":    away.get("team", {}).get("name", "Unknown"),
                "home_team":    home.get("team", {}).get("name", "Unknown"),
                "away_id":      away.get("team", {}).get("id"),
                "home_id":      home.get("team", {}).get("id"),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "venue":        g.get("venue", {}).get("name", ""),
                "time":         time_str,
            })
    return games

# ── Fetch MLB hitting stats ────────────────────────────────────────────────────
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
        hr = int(stat.get("homeRuns", 0) or 0)

        def sf(key):
            v = stat.get(key)
            try:
                return float(v) if v not in (None, ".---", "-.--") else 0.0
            except (ValueError, TypeError):
                return 0.0

        avg = sf("avg")
        slg = sf("sluggingPercentage")
        iso = max(round(slg - avg, 3), 0.0)   # clamp to 0 — no negatives

        rows.append({
            "player_id": player.get("id"),
            "Player":    player.get("fullName", "Unknown"),
            "team_id":   team.get("id"),
            "Team":      team.get("name", "Unknown"),
            "PA": pa, "HR": hr, "ISO": iso,
        })
    return pd.DataFrame(rows)

# ── Fetch Statcast main (Savant) ───────────────────────────────────────────────
# Confirmed column names: avg_hit_speed, avg_hit_angle, ev95percent,
#                         anglesweetspotpercent, fbld
@st.cache_data(ttl=3600)
def fetch_statcast_main(season: int) -> pd.DataFrame:
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&id=&sportId=1"
        f"&season={season}&season_end={season}&min=50&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if not name_col:
        return pd.DataFrame()
    df["Player"] = df[name_col].apply(parse_savant_name)

    rename = {
        "avg_hit_speed":         "Avg EV",
        "avg_hit_angle":         "Avg LA",
        "ev95percent":           "HH%",
        "anglesweetspotpercent": "SweetSpot%",
        "fbld":                  "FB%",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Player"] + [c for c in rename.values() if c in df.columns]
    df   = df[keep].copy()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Player"])

# ── Fetch Statcast barrels (Savant) ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_statcast_barrels(season: int) -> pd.DataFrame:
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&id=barrel_batted_rate&sportId=1"
        f"&season={season}&season_end={season}&min=50&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if not name_col:
        return pd.DataFrame()
    df["Player"] = df[name_col].apply(parse_savant_name)

    brl_col = next(
        (c for c in df.columns if c.lower() in ("brl_percent", "barrel_batted_rate", "brl_pa")),
        None
    )
    rename = {}
    if brl_col:
        rename[brl_col] = "Brl/BIP%"

    df = df.rename(columns=rename)
    keep = ["Player"] + [c for c in ["Brl/BIP%"] if c in df.columns]
    if len(keep) == 1:
        return pd.DataFrame()
    df = df[keep].copy()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Player"])

# ── Fetch SwStr% from FanGraphs ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fangraphs_swstr(season: int) -> pd.DataFrame:
    """
    FanGraphs leaderboard export — batter plate discipline.
    SwStr% = swinging strike rate (swings and misses / total pitches)
    """
    url = (
        f"https://www.fangraphs.com/leaders/major-league"
        f"?pos=all&stats=bat&lg=all&qual=50&type=5"
        f"&season={season}&season1={season}&ind=0&csv=1"
    )
    try:
        r = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.fangraphs.com/",
        })
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    # FanGraphs name column is just "Name"
    if "Name" not in df.columns:
        return pd.DataFrame()

    df["Player"] = df["Name"].str.strip()

    # SwStr% column varies — try common names
    swstr_col = next(
        (c for c in df.columns if c.lower() in ("swstr%", "swstr", "sw_str%", "swinging_strike%")),
        next((c for c in df.columns if "swstr" in c.lower()), None)
    )
    if not swstr_col:
        return pd.DataFrame()

    df = df.rename(columns={swstr_col: "SwStr%"})
    df["SwStr%"] = pd.to_numeric(
        df["SwStr%"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    # FanGraphs returns decimals like 0.112 — convert to percent if needed
    if df["SwStr%"].dropna().max() < 1.0:
        df["SwStr%"] = df["SwStr%"] * 100

    return df[["Player", "SwStr%"]].dropna(subset=["Player"])

# ── Fetch PulledBrl% from FanGraphs ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fangraphs_pulledbrl(season: int) -> pd.DataFrame:
    """
    FanGraphs batted ball leaderboard — type=2 includes pull%, brl stats.
    """
    url = (
        f"https://www.fangraphs.com/leaders/major-league"
        f"?pos=all&stats=bat&lg=all&qual=50&type=2"
        f"&season={season}&season1={season}&ind=0&csv=1"
    )
    try:
        r = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.fangraphs.com/",
        })
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if "Name" not in df.columns:
        return pd.DataFrame()

    df["Player"] = df["Name"].str.strip()

    pulled_col = next(
        (c for c in df.columns if "pull" in c.lower() and "brl" in c.lower()),
        next((c for c in df.columns if "pulledbrl" in c.lower().replace(" ", "").replace("_", "")), None)
    )
    if not pulled_col:
        return pd.DataFrame()

    df = df.rename(columns={pulled_col: "PulledBrl%"})
    df["PulledBrl%"] = pd.to_numeric(
        df["PulledBrl%"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    if df["PulledBrl%"].dropna().max() < 1.0:
        df["PulledBrl%"] = df["PulledBrl%"] * 100

    return df[["Player", "PulledBrl%"]].dropna(subset=["Player"])

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

# ── 0-100 matchup score ────────────────────────────────────────────────────────
# Calculated AFTER building the full slate so scores are relative to each other.
# Formula components (all scaled 0–1 then weighted):
#   HH%       weight 25% — hard contact rate
#   Brl/BIP%  weight 25% — barrel rate
#   ISO       weight 20% — raw power
#   SweetSpot% weight 10% — launch angle quality
#   Park factor weight 10% — venue boost
#   Pitcher HR/9 weight 10% — matchup vulnerability
# SwStr% is INVERTED (lower = better) and not in score — it's a display stat only

SCORE_WEIGHTS = {
    "HH%":          0.25,
    "Brl/BIP%":     0.25,
    "ISO":          0.20,
    "SweetSpot%":   0.10,
    "Park factor":  0.10,
    "Pitcher HR/9": 0.10,
}

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each component 0–1 relative to today's slate, apply weights → 0–100 score."""
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    for col, weight in SCORE_WEIGHTS.items():
        if col not in df.columns or df[col].isna().all():
            continue
        col_vals = df[col].fillna(df[col].median())
        min_v, max_v = col_vals.min(), col_vals.max()
        if max_v == min_v:
            scaled = pd.Series(0.5, index=df.index)
        else:
            scaled = (col_vals - min_v) / (max_v - min_v)
        score += scaled * weight

    # Normalize to 0–100
    s_min, s_max = score.min(), score.max()
    if s_max > s_min:
        df["Matchup score"] = ((score - s_min) / (s_max - s_min) * 100).round(1)
    else:
        df["Matchup score"] = 50.0
    return df

# ── Build raw matchup rows (no score yet) ─────────────────────────────────────
STATCAST_COLS = ["HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%",
                 "SwStr%", "Brl/BIP%", "PulledBrl%"]

def build_raw_rows(batting_id, batting_team, opp_pitcher_name, home_team,
                   hitting_df, pitcher_df, min_pa, min_hr):
    batters = hitting_df[
        (hitting_df["team_id"] == batting_id) &
        (hitting_df["PA"] >= min_pa) &
        (hitting_df["HR"] >= min_hr)
    ]
    if batters.empty:
        return []

    park_factor  = PARK_FACTORS.get(home_team, 1.00)
    pr           = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher_name]
    opp_hr9      = pr.iloc[0]["HR/9"] if not pr.empty else None

    rows = []
    for _, b in batters.iterrows():
        row = {
            "Player":       b["Player"],
            "Batting team": batting_team,
            "Opp pitcher":  opp_pitcher_name,
            "HR":           b["HR"],
            "ISO":          b["ISO"],
            "Park factor":  park_factor,
            "Pitcher HR/9": opp_hr9,
        }
        for col in STATCAST_COLS:
            row[col] = b.get(col, None)
        rows.append(row)
    return rows

# ── Display columns ────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "Player", "Batting team", "Opp pitcher",
    "HR",
    "HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%",
    "SwStr%", "Brl/BIP%", "PulledBrl%",
    "ISO",
    "Park factor", "Pitcher HR/9",
    "Matchup score",
]

HIGH_GOOD = ["HH%", "Avg EV", "FB%", "SweetSpot%", "Brl/BIP%", "PulledBrl%",
             "ISO", "Park factor", "Pitcher HR/9", "Matchup score"]
LOW_GOOD  = ["SwStr%"]

def style_table(df: pd.DataFrame, cols: list):
    fmt = {
        "HH%":          "{:.1f}%",
        "Avg EV":       "{:.1f}",
        "Avg LA":       "{:.1f}°",
        "FB%":          "{:.1f}%",
        "SweetSpot%":   "{:.1f}%",
        "SwStr%":       "{:.1f}%",
        "Brl/BIP%":     "{:.1f}%",
        "PulledBrl%":   "{:.1f}%",
        "ISO":          "{:.3f}",
        "Park factor":  "{:.2f}",
        "Pitcher HR/9": "{:.2f}",
        "Matchup score":"{:.1f}",
    }
    fmt    = {k: v for k, v in fmt.items() if k in cols}
    styled = df[cols].style.format(fmt, na_rep="—")
    for col in HIGH_GOOD:
        if col in cols and df[col].notna().any():
            styled = styled.background_gradient(subset=[col], cmap="RdYlGn")
    for col in LOW_GOOD:
        if col in cols and df[col].notna().any():
            styled = styled.background_gradient(subset=[col], cmap="RdYlGn_r")
    return styled

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    season        = st.selectbox("Season", [2026, 2025, 2024], index=0)
    selected_date = st.date_input("Game date", value=date.today())
    min_pa        = st.slider("Min plate appearances", 30, 300, 80, step=10)
    min_hr        = st.slider("Min HR this season",     0,  20,  3, step=1)
    st.markdown("---")
    st.markdown("**Colour guide**")
    st.markdown("🟢 Green = best · 🟡 Yellow = average · 🔴 Red = bad")
    st.caption("SwStr% reversed — green = low whiffs (good contact).")
    st.markdown("---")
    st.markdown("**Matchup score (0–100)**")
    st.caption("Scores are relative to today's slate — 100 = best matchup of the day.")
    st.caption("Weighted formula:")
    st.caption("• HH% — 25%")
    st.caption("• Brl/BIP% — 25%")
    st.caption("• ISO — 20%")
    st.caption("• SweetSpot% — 10%")
    st.caption("• Park factor — 10%")
    st.caption("• Pitcher HR/9 — 10%")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.caption("MLB Stats API — HR, ISO, schedule, pitchers")
    st.caption("Baseball Savant — HH%, EV, LA, FB%, SweetSpot%, Brl/BIP%")
    st.caption("FanGraphs — SwStr%, PulledBrl%")

# ── Load data ──────────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh hourly · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule, MLB stats, Statcast, and FanGraphs data..."):
    games      = fetch_schedule(date_str)
    hitting_df = fetch_hitting_stats(season)
    sc_main    = fetch_statcast_main(season)
    sc_barrels = fetch_statcast_barrels(season)
    fg_swstr   = fetch_fangraphs_swstr(season)
    fg_pulled  = fetch_fangraphs_pulledbrl(season)
    pitcher_df = fetch_pitcher_stats(season)

if not games:
    st.warning(f"No games found for {date_str}. Try a different date.")
    st.stop()
if hitting_df.empty:
    st.warning("Could not load hitting stats. Try again in a moment.")
    st.stop()

# ── Merge all sources ──────────────────────────────────────────────────────────
for src_df in [sc_main, sc_barrels, fg_swstr, fg_pulled]:
    if not src_df.empty:
        hitting_df = hitting_df.merge(src_df, on="Player", how="left")

for col in STATCAST_COLS:
    if col not in hitting_df.columns:
        hitting_df[col] = None

# Source status banner
sources_ok  = []
sources_bad = []
(sources_ok if not sc_main.empty    else sources_bad).append("Savant (EV/HH%/LA)")
(sources_ok if not sc_barrels.empty else sources_bad).append("Savant (Brl/BIP%)")
(sources_ok if not fg_swstr.empty   else sources_bad).append("FanGraphs (SwStr%)")
(sources_ok if not fg_pulled.empty  else sources_bad).append("FanGraphs (PulledBrl%)")
if sources_bad:
    st.warning(f"⚠️ Could not load: {', '.join(sources_bad)} — those columns show '—'.")

# ── Game selector ──────────────────────────────────────────────────────────────
st.subheader("Filter by game")
all_label = f"All games today ({len(games)} games)"
options   = [all_label] + [
    f"{g['away_team']} @ {g['home_team']}  —  {g['time']}" for g in games
]
choice = st.selectbox("", options, label_visibility="collapsed")
selected_games = (
    games if choice == all_label
    else [g for g in games
          if f"{g['away_team']} @ {g['home_team']}  —  {g['time']}" == choice]
)

# ── Build ALL rows first so scores are relative to full slate ─────────────────
all_raw = []
game_map = {}  # track which rows belong to which game+team

for g in selected_games:
    for batting_id, batting_team, opp_pitcher, label_key in [
        (g["away_id"], g["away_team"], g["home_pitcher"], f"{g['away_team']}__{g['away_team']}@{g['home_team']}"),
        (g["home_id"], g["home_team"], g["away_pitcher"], f"{g['home_team']}__{g['away_team']}@{g['home_team']}"),
    ]:
        rows = build_raw_rows(batting_id, batting_team, opp_pitcher, g["home_team"],
                              hitting_df, pitcher_df, min_pa, min_hr)
        for row in rows:
            row["_key"] = label_key
        all_raw.extend(rows)

if not all_raw:
    st.info("No batters match your filters. Try lowering Min PA or Min HR.")
    st.stop()

# Score all rows together (0–100 relative to today's slate)
full_df = compute_scores(pd.DataFrame(all_raw))

# ── Render each game — teams stacked vertically ────────────────────────────────
for g in selected_games:
    pf       = PARK_FACTORS.get(g["home_team"], 1.00)
    pf_emoji = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    pf_label = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral park"

    st.markdown("---")
    st.markdown(f"### ⚾ {g['away_team']} @ {g['home_team']}")
    st.caption(f"{g['time']} · {g['venue']} · {pf_emoji} {pf_label} (PF {pf:.2f})")

    for batting_team, border_color, opp_pitcher, key in [
        (g["away_team"], "var(--color-border-info)",    g["home_pitcher"],
         f"{g['away_team']}__{g['away_team']}@{g['home_team']}"),
        (g["home_team"], "var(--color-border-success)", g["away_pitcher"],
         f"{g['home_team']}__{g['away_team']}@{g['home_team']}"),
    ]:
        st.markdown(
            f"<div style='background:var(--color-background-secondary);"
            f"border-left:3px solid {border_color};"
            f"padding:8px 14px;border-radius:var(--border-radius-md);margin:10px 0 4px'>"
            f"<span style='font-size:14px;font-weight:500'>{batting_team}</span>"
            f"<span style='font-size:12px;color:var(--color-text-secondary)'>"
            f" batting vs {opp_pitcher}</span></div>",
            unsafe_allow_html=True,
        )

        team_df = (
            full_df[full_df["_key"] == key]
            .drop(columns=["_key"])
            .sort_values("Matchup score", ascending=False)
            .reset_index(drop=True)
        )
        team_df.index += 1

        if team_df.empty:
            st.info(f"No {batting_team} batters match current filters.")
        else:
            valid = [c for c in DISPLAY_COLS if c in team_df.columns]
            st.dataframe(style_table(team_df, valid), use_container_width=True, height=380)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Top picks chart ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top picks across today's slate")

chart_df = (
    full_df.drop(columns=["_key"], errors="ignore")
    .sort_values("Matchup score", ascending=False)
    .reset_index(drop=True)
)
chart_df.index += 1

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Top 15 matchup scores**")
    top15 = chart_df.head(15).sort_values("Matchup score")
    fig = px.bar(
        top15, x="Matchup score", y="Player", orientation="h",
        color="Matchup score", color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        hover_data=["Batting team", "Opp pitcher", "HR", "HH%", "Brl/BIP%"],
        text="Opp pitcher",
    )
    fig.update_traces(textposition="inside", textfont_size=9)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=440,
        yaxis_title="", coloraxis_showscale=False,
        xaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Brl/BIP% vs HH% — today's batters**")
    plot_df = chart_df.head(60)
    hh_ok  = "HH%"      in plot_df.columns and plot_df["HH%"].notna().any()
    brl_ok = "Brl/BIP%" in plot_df.columns and plot_df["Brl/BIP%"].notna().any()
    if hh_ok and brl_ok:
        fig2 = px.scatter(
            plot_df, x="HH%", y="Brl/BIP%", text="Player",
            color="Matchup score", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            size="HR",
            hover_data=["Batting team", "Opp pitcher", "ISO", "Avg EV", "SweetSpot%"],
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=440)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Statcast data unavailable from Savant today.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API · Baseball Savant Statcast · FanGraphs · "
    "Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
