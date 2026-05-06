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
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Player"] + [c for c in rename.values() if c in df.columns]
    df   = df[keep].copy()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Player"])

# ── Savant: FB% from exit velocity spray leaderboard ─────────────────────────
@st.cache_data(ttl=3600)
def fetch_savant_fb(season: int) -> pd.DataFrame:
    """
    Savant's zone/spray leaderboard has true fly ball rate.
    Endpoint confirmed to include batted ball type breakdowns.
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&id=n_fb_percent&sportId=1"
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

    # Find any column that looks like fly ball rate
    # Savant confirmed cols from this endpoint include: gb, fbld, n_fb_percent
    fb_col = next(
        (c for c in df.columns
         if c.lower() in ("n_fb_percent", "fb_percent", "fly_ball_percent", "fb", "fly_ball")),
        None
    )

    if fb_col is None:
        # Last resort: if gb and fbld exist, ld% ≈ 100 - gb - fbld won't work
        # but if there's a column with values in realistic FB% range (20-50) use it
        for c in df.columns:
            try:
                col_vals = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(col_vals) > 10 and 15 < col_vals.mean() < 50:
                    fb_col = c
                    break
            except Exception:
                continue

    if fb_col is None:
        return pd.DataFrame()

    df = df.rename(columns={fb_col: "FB%"})
    df = df[["Player", "FB%"]].copy()
    df["FB%"] = pd.to_numeric(df["FB%"], errors="coerce")
    if df["FB%"].dropna().max() < 1.0:
        df["FB%"] = df["FB%"] * 100
    # Sanity check — real FB% is 20-50%, discard if out of range
    median = df["FB%"].dropna().median()
    if not (15 < median < 55):
        return pd.DataFrame()
    return df.dropna(subset=["Player"])

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

# ── Last 3 games EV, LA, FB% from MLB game log API ───────────────────────────
@st.cache_data(ttl=1800)
def fetch_last3_ev(team_id: int, season: int) -> pd.DataFrame:
    """
    Pulls each player's game log for the season, takes last 3 games,
    aggregates avg exit velocity, avg launch angle, and fly ball rate.
    MLB API returns per-game hitting stats including batted ball data.
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
        games = data["games"][-3:]
        if not games:
            continue

        # Exit velocity and launch angle from game log
        ev_vals, la_vals = [], []
        fly_balls, total_batted = 0, 0

        for g in games:
            ev = g.get("avgExitVelocity") or g.get("launchSpeed")
            la = g.get("avgLaunchAngle") or g.get("launchAngle")
            fb = int(g.get("flyOuts", 0) or g.get("airOuts", 0) or 0)
            ab = int(g.get("atBats", 0) or 0)

            if ev is not None:
                try: ev_vals.append(float(ev))
                except (ValueError, TypeError): pass
            if la is not None:
                try: la_vals.append(float(la))
                except (ValueError, TypeError): pass
            fly_balls    += fb
            total_batted += ab

        avg_ev = round(sum(ev_vals) / len(ev_vals), 1) if ev_vals else None
        avg_la = round(sum(la_vals) / len(la_vals), 1) if la_vals else None
        fb_pct = round(fly_balls / total_batted * 100, 1) if total_batted > 0 else None

        rows.append({
            "player_id":    pid,
            "Player":       data["name"],
            "Avg EV (L3G)": avg_ev,
            "Avg LA (L3G)": avg_la,
            "FB% (L3G)":    fb_pct,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()

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
    "Avg EV (L3G)": 0.55,
    "HH%":          0.15,
    "Avg LA (L3G)": 0.10,
    "FB% (L3G)":    0.10,
    "Pitcher HR/9": 0.05,
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
STATCAST_COLS = ["HH%", "Avg EV", "Avg LA", "FB%"]

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
            "Player":         b["Player"],
            "Batting team":   batting_team,
            "Opp pitcher":    opp_pitcher,
            "HR":             b["HR"],
            "AB":             b["AB"],
            "SLG":            b["SLG"],
            "Park factor":    park_factor,
            "Pitcher HR/9":   opp_hr9,
            "HH%":            b.get("HH%", None),
            "Avg EV (L3G)":   b.get("Avg EV (L3G)", None),
            "Avg LA (L3G)":   b.get("Avg LA (L3G)", None),
            "FB% (L3G)":      b.get("FB% (L3G)", None),
        }
        rows.append(row)
    return rows

# ── Display columns ────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "Player", "Batting team", "Opp pitcher",
    "HR", "AB",
    "HH%",
    "Avg EV (L3G)", "Avg LA (L3G)", "FB% (L3G)",
    "SLG",
    "Park factor", "Pitcher HR/9",
    "Matchup score",
]

HIGH_GOOD = ["HH%", "Avg EV (L3G)", "FB% (L3G)",
             "SLG", "Park factor", "Pitcher HR/9", "Matchup score"]

def style_table(df: pd.DataFrame, cols: list):
    fmt = {
        "HH%":           "{:.1f}%",
        "Avg EV (L3G)":  "{:.1f}",
        "Avg LA (L3G)":  "{:.1f}°",
        "FB% (L3G)":     "{:.1f}%",
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
    st.caption("• Avg EV (last 3 games) — 55%")
    st.caption("• HH% — 15%")
    st.caption("• Avg LA (last 3 games) — 10%")
    st.caption("• FB% (last 3 games) — 10%")
    st.caption("• Pitcher HR/9 — 5%")
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

# ── Merge full-season Statcast ─────────────────────────────────────────────────
for src in [sc_main, sc_barrels, sc_fb]:
    if not src.empty:
        hitting_df = hitting_df.merge(src, on="Player", how="left")

for col in STATCAST_COLS:
    if col not in hitting_df.columns:
        hitting_df[col] = None

# ── Fetch and merge last 3 games EV/LA/FB% per team ──────────────────────────
team_ids = set()
for g in games:
    if g["away_id"]: team_ids.add((g["away_id"], g["away_team"]))
    if g["home_id"]: team_ids.add((g["home_id"], g["home_team"]))

with st.spinner("Loading last 3 games exit velocity data..."):
    l3g_frames = []
    for tid, _ in team_ids:
        df_t = fetch_last3_ev(tid, season)
        if not df_t.empty:
            l3g_frames.append(df_t)

if l3g_frames:
    l3g_df = pd.concat(l3g_frames).drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    # Merge by player_id for accuracy (avoids name mismatches)
    if "player_id" in hitting_df.columns:
        hitting_df = hitting_df.merge(
            l3g_df[["player_id", "Avg EV (L3G)", "Avg LA (L3G)", "FB% (L3G)"]],
            on="player_id", how="left"
        )
    else:
        hitting_df = hitting_df.merge(
            l3g_df[["Player", "Avg EV (L3G)", "Avg LA (L3G)", "FB% (L3G)"]],
            on="Player", how="left"
        )
else:
    hitting_df["Avg EV (L3G)"] = None
    hitting_df["Avg LA (L3G)"] = None
    hitting_df["FB% (L3G)"]    = None

# Status banner
failed = []
if sc_main.empty: failed.append("HH%")
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
        hover_data=["Batting team", "Opp pitcher", "HR", "HH%"],
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
    st.markdown("**Avg EV vs HH% — today's batters (last 3 games)**")
    plot_df = chart_df.head(60)
    hh_ok = "HH%"           in plot_df.columns and plot_df["HH%"].notna().any()
    ev_ok = "Avg EV (L3G)"  in plot_df.columns and plot_df["Avg EV (L3G)"].notna().any()
    if hh_ok and ev_ok:
        fig2 = px.scatter(
            plot_df, x="HH%", y="Avg EV (L3G)", text="Player",
            color="Matchup score", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            size="HR",
            hover_data=["Batting team", "Opp pitcher", "SLG", "Avg LA (L3G)", "FB% (L3G)"],
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=440)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("EV data unavailable — chart will appear once last 3 games data loads.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API + Baseball Savant Statcast · "
    "Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
