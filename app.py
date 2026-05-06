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

# ── Savant name parser ─────────────────────────────────────────────────────────
def parse_savant_name(val: str) -> str:
    if "," in str(val):
        parts = [p.strip() for p in str(val).split(",")]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return str(val).strip()

def add_player_col(df: pd.DataFrame) -> pd.DataFrame:
    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if name_col:
        df["Player"] = df[name_col].apply(parse_savant_name)
    return df

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

# ── Fetch full-season hitting stats ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_season_stats(season: int) -> pd.DataFrame:
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
        ab  = int(stat.get("atBats", 0) or 0)
        hr  = int(stat.get("homeRuns", 0) or 0)
        tb  = int(stat.get("totalBases", 0) or 0)
        slg = round(tb / ab, 3) if ab > 0 else None

        rows.append({
            "player_id": player.get("id"),
            "Player":    player.get("fullName", "Unknown"),
            "team_id":   team.get("id"),
            "Team":      team.get("name", "Unknown"),
            "AB": ab,
            "HR": hr,
            "SLG": slg,
        })
    return pd.DataFrame(rows)

# ── Fetch last N games stats for ALL players on a team ────────────────────────
@st.cache_data(ttl=1800)
def fetch_last_n_games(team_id: int, season: int, n_games: int) -> pd.DataFrame:
    """
    Pulls each player's game log for the season, keeps last N games,
    aggregates AB, HR, SLG for that window.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=gameLog&group=hitting&gameType=R"
        f"&season={season}&teamId={team_id}&limit=9999"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        splits = r.json()["stats"][0]["splits"]
    except Exception:
        return pd.DataFrame()

    # Group by player, take last N game entries
    player_games: dict = {}
    for s in splits:
        stat   = s.get("stat", {})
        player = s.get("player", {})
        pid    = player.get("id")
        name   = player.get("fullName", "Unknown")
        if pid not in player_games:
            player_games[pid] = {"name": name, "games": []}
        player_games[pid]["games"].append(stat)

    rows = []
    for pid, data in player_games.items():
        games = data["games"][-n_games:]   # last N games only
        if not games:
            continue
        ab  = sum(int(g.get("atBats", 0) or 0) for g in games)
        hr  = sum(int(g.get("homeRuns", 0) or 0) for g in games)
        tb  = sum(int(g.get("totalBases", 0) or 0) for g in games)
        slg = round(tb / ab, 3) if ab > 0 else 0.0
        rows.append({
            "player_id": pid,
            "Player":    data["name"],
            "AB":        ab,
            "HR":        hr,
            "SLG":       slg,
        })
    return pd.DataFrame(rows)

# ── Savant: main leaderboard — HH%, Avg EV, Avg LA, SweetSpot% ───────────────
@st.cache_data(ttl=3600)
def fetch_savant_main(season: int) -> pd.DataFrame:
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

    df = add_player_col(df)
    if "Player" not in df.columns:
        return pd.DataFrame()

    rename = {
        "avg_hit_speed":         "Avg EV",
        "avg_hit_angle":         "Avg LA",
        "ev95percent":           "HH%",
        "anglesweetspotpercent": "SweetSpot%",
        "gb":                    "_gb",
        "fbld":                  "_fbld",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Player"] + [c for c in rename.values() if c in df.columns]
    df   = df[keep].copy()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive FB% from fbld - league avg LD% (~21%), clipped to sensible range
    if "_fbld" in df.columns:
        df["FB%"] = (df["_fbld"] - 21.0).clip(lower=10.0, upper=55.0)
        df = df.drop(columns=["_gb", "_fbld"], errors="ignore")
        keep = [c for c in keep if c not in ("_gb", "_fbld")] + ["FB%"]

    return df.dropna(subset=["Player"])

# ── FB% is now derived inside fetch_savant_main from fbld - avg LD% ───────────
def fetch_savant_fb(season: int) -> pd.DataFrame:
    return pd.DataFrame()  # no longer needed as a separate call

# ── Savant: barrel leaderboard — Brl/BIP% ─────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_savant_barrels(season: int) -> pd.DataFrame:
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

    df = add_player_col(df)
    if "Player" not in df.columns:
        return pd.DataFrame()

    brl_col = next(
        (c for c in df.columns
         if c.lower() in ("brl_percent", "brl_pa", "barrel_batted_rate")),
        None
    )
    if not brl_col:
        return pd.DataFrame()

    df = df.rename(columns={brl_col: "Brl/BIP%"})
    df = df[["Player", "Brl/BIP%"]].copy()
    df["Brl/BIP%"] = pd.to_numeric(df["Brl/BIP%"], errors="coerce")
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

# ── 0-100 matchup score ────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "HH%":          0.25,
    "Brl/BIP%":     0.25,
    "Avg EV":       0.15,
    "Avg LA":       0.10,
    "FB%":          0.10,
    "Pitcher HR/9": 0.10,
    "SweetSpot%":   0.05,
    "Park factor":  0.05,
}

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    score = pd.Series(0.0, index=df.index)
    for col, weight in SCORE_WEIGHTS.items():
        if col not in df.columns or df[col].isna().all():
            continue
        vals   = df[col].fillna(df[col].median())
        lo, hi = vals.min(), vals.max()
        scaled = (vals - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=df.index)
        score += scaled * weight
    s_min, s_max = score.min(), score.max()
    df["Matchup score"] = (
        ((score - s_min) / (s_max - s_min) * 100).round(1)
        if s_max > s_min else pd.Series(50.0, index=df.index)
    )
    return df

# ── Build raw rows ─────────────────────────────────────────────────────────────
STATCAST_COLS = ["HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%", "Brl/BIP%"]

def build_raw_rows(batting_id, batting_team, opp_pitcher, home_team,
                   hitting_df, pitcher_df, min_ab, min_hr):
    batters = hitting_df[
        (hitting_df["team_id"] == batting_id) &
        (hitting_df["AB"] >= min_ab) &
        (hitting_df["HR"] >= min_hr)
    ]
    if batters.empty:
        return []

    park_factor = PARK_FACTORS.get(home_team, 1.00)
    pr          = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher]
    opp_hr9     = pr.iloc[0]["HR/9"] if not pr.empty else None

    rows = []
    for _, b in batters.iterrows():
        row = {
            "Player":       b["Player"],
            "Batting team": batting_team,
            "Opp pitcher":  opp_pitcher,
            "HR":           b["HR"],
            "AB":           b["AB"],
            "SLG":          b["SLG"],
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
    "HR", "AB",
    "HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%", "Brl/BIP%",
    "SLG",
    "Park factor", "Pitcher HR/9",
    "Matchup score",
]

HIGH_GOOD = ["HH%", "Avg EV", "FB%", "SweetSpot%", "Brl/BIP%",
             "SLG", "Park factor", "Pitcher HR/9", "Matchup score"]

def style_table(df: pd.DataFrame, cols: list):
    fmt = {
        "HH%":           "{:.1f}%",
        "Avg EV":        "{:.1f}",
        "Avg LA":        "{:.1f}°",
        "FB%":           "{:.1f}%",
        "SweetSpot%":    "{:.1f}%",
        "Brl/BIP%":      "{:.1f}%",
        "SLG":           "{:.3f}",
        "Park factor":   "{:.2f}",
        "Pitcher HR/9":  "{:.2f}",
        "Matchup score": "{:.1f}",
    }
    fmt    = {k: v for k, v in fmt.items() if k in cols}
    styled = df[cols].style.format(fmt, na_rep="—")
    for col in HIGH_GOOD:
        if col in cols and df[col].notna().any():
            styled = styled.background_gradient(subset=[col], cmap="RdYlGn")
    return styled

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    season        = st.selectbox("Season", [2026, 2025, 2024], index=0)
    selected_date = st.date_input("Game date", value=date.today())

    st.markdown("---")
    st.markdown("**Minimum filters**")
    min_ab = st.slider("Min at bats", 0, 100, 20, step=5)
    min_hr = st.slider("Min HR",      0,  20,  0, step=1)

    st.markdown("---")
    st.markdown("**Colour guide**")
    st.markdown("🟢 Green = best · 🟡 Yellow = average · 🔴 Red = bad")
    st.markdown("---")
    st.markdown("**Matchup score (0–100)**")
    st.caption("100 = best matchup on today's slate.")
    st.caption("• HH% — 25%")
    st.caption("• Brl/BIP% — 25%")
    st.caption("• Avg EV — 15%")
    st.caption("• Avg LA — 10%")
    st.caption("• FB% — 10%")
    st.caption("• Pitcher HR/9 — 10%")
    st.caption("• SweetSpot% — 5%")
    st.caption("• Park factor — 5%")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.caption("MLB Stats API — HR, SLG, schedule, pitchers")
    st.caption("Baseball Savant — HH%, EV, LA, FB%, SweetSpot%, Brl/BIP%")

# ── Load base data ─────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh hourly · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule, Statcast, and pitcher data..."):
    games      = fetch_schedule(date_str)
    sc_main    = fetch_savant_main(season)
    sc_barrels = fetch_savant_barrels(season)
    sc_fb      = fetch_savant_fb(season)
    pitcher_df = fetch_pitcher_stats(season)

if not games:
    st.warning(f"No games found for {date_str}. Try a different date.")
    st.stop()

with st.spinner("Loading full season hitting stats..."):
    hitting_df = fetch_season_stats(season)

if hitting_df.empty:
    st.warning("Could not load hitting stats. Try again in a moment.")
    st.stop()

# ── Merge Statcast (always full-season — Savant doesn't do recent splits) ─────
for src in [sc_main, sc_barrels, sc_fb]:
    if not src.empty:
        hitting_df = hitting_df.merge(src, on="Player", how="left")

for col in STATCAST_COLS:
    if col not in hitting_df.columns:
        hitting_df[col] = None

# Status banner
failed = []
if sc_main.empty:    failed.append("EV / HH% / LA / FB% / SweetSpot%")
if sc_barrels.empty: failed.append("Brl/BIP%")
if failed:
    st.warning(f"⚠️ Could not load from Savant: {', '.join(failed)} — those columns show '—'.")

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

# ── Build all rows and score ───────────────────────────────────────────────────
all_raw = []
for g in selected_games:
    for batting_id, batting_team, opp_pitcher, key in [
        (g["away_id"], g["away_team"], g["home_pitcher"],
         f"{g['away_team']}__{g['away_team']}@{g['home_team']}"),
        (g["home_id"], g["home_team"], g["away_pitcher"],
         f"{g['home_team']}__{g['away_team']}@{g['home_team']}"),
    ]:
        rows = build_raw_rows(batting_id, batting_team, opp_pitcher, g["home_team"],
                              hitting_df, pitcher_df, min_ab, min_hr)
        for row in rows:
            row["_key"] = key
        all_raw.extend(rows)

if not all_raw:
    st.info("No batters match your filters. Try lowering Min AB or Min HR.")
    st.stop()

full_df = compute_scores(pd.DataFrame(all_raw))

# ── Render each game — teams stacked vertically ────────────────────────────────
for g in selected_games:
    pf       = PARK_FACTORS.get(g["home_team"], 1.00)
    pf_emoji = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    pf_label = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral park"

    st.markdown("---")
    st.markdown(f"### ⚾ {g['away_team']} @ {g['home_team']}")
    st.caption(f"{g['time']} · {g['venue']} · {pf_emoji} {pf_label} (PF {pf:.2f})")

    for batting_team, border, opp_pitcher, key in [
        (g["away_team"], "var(--color-border-info)",    g["home_pitcher"],
         f"{g['away_team']}__{g['away_team']}@{g['home_team']}"),
        (g["home_team"], "var(--color-border-success)", g["away_pitcher"],
         f"{g['home_team']}__{g['away_team']}@{g['home_team']}"),
    ]:
        st.markdown(
            f"<div style='background:var(--color-background-secondary);"
            f"border-left:3px solid {border};"
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
            hover_data=["Batting team", "Opp pitcher", "SLG", "Avg EV", "SweetSpot%"],
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=440)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Statcast data unavailable from Savant today.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API + Baseball Savant Statcast · "
    "Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
