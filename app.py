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
        "avg_hit_speed": "Avg EV",
        "ev95percent":   "HH%",
        "brl_percent":   "Barrel%",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Player"] + [c for c in ["Avg EV", "HH%", "Barrel%"] if c in df.columns]
    df   = df[keep].copy()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Player"])

# ── Savant: SwStr% from whiff leaderboard ────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_savant_swstr(season: int) -> pd.DataFrame:
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&id=whiff_percent&sportId=1"
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

    whiff_col = next(
        (c for c in df.columns if c.lower() in ("whiff_percent", "whiff_pct")),
        next((c for c in df.columns if "whiff" in c.lower()), None)
    )
    if not whiff_col:
        return pd.DataFrame()

    df = df.rename(columns={whiff_col: "SwStr%"})
    df["SwStr%"] = pd.to_numeric(df["SwStr%"], errors="coerce")
    if df["SwStr%"].dropna().max() < 1.0:
        df["SwStr%"] = df["SwStr%"] * 100
    return df[["Player", "SwStr%"]].dropna(subset=["Player"])

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

# ── Savant: barrel leaderboard — Barrel% ─────────────────────────────────────
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

    # This endpoint has confirmed cols: brl_percent, brl_pa, barrels
    # Name col is "last_name, first_name" (single col with comma)
    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if not name_col:
        return pd.DataFrame()
    df["Player"] = df[name_col].apply(parse_savant_name)

    brl_col = next(
        (c for c in df.columns if c.lower() in ("brl_percent", "brl_pa", "barrel_batted_rate")),
        None
    )
    if not brl_col:
        return pd.DataFrame()

    df = df.rename(columns={brl_col: "Barrel%"})
    df = df[["Player", "Barrel%"]].copy()
    df["Barrel%"] = pd.to_numeric(df["Barrel%"], errors="coerce")
    return df.dropna(subset=["Player"])

# ── Last 3 games FB% from MLB game log, EV/LA from Savant season leaderboard ──
# Note: MLB Stats API game logs do not include exit velocity or launch angle.
# FB% is computed from airOuts/atBats from each player's individual game log.
@st.cache_data(ttl=1800)
def fetch_last3_fb(team_id: int, season: int) -> pd.DataFrame:
    """
    1. Gets active roster for the team
    2. For each player, fetches their game log and takes last 3 games
    3. Computes FB% = airOuts / atBats * 100
    """
    # Step 1 — get roster
    roster_url = (
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        f"?rosterType=active&season={season}"
    )
    try:
        r = requests.get(roster_url, timeout=10)
        r.raise_for_status()
        players = r.json().get("roster", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for p in players:
        pid  = p.get("person", {}).get("id")
        name = p.get("person", {}).get("fullName", "Unknown")
        if not pid:
            continue

        # Step 2 — get this player's game log
        log_url = (
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=gameLog&group=hitting&gameType=R&season={season}"
        )
        try:
            r2 = requests.get(log_url, timeout=10)
            r2.raise_for_status()
            splits = r2.json()["stats"][0]["splits"]
        except Exception:
            continue

        if not splits:
            continue

        # Step 3 — last 3 games
        last3 = splits[-3:]
        fly_balls    = sum(int(s.get("stat", {}).get("airOuts", 0) or 0) for s in last3)
        total_batted = sum(int(s.get("stat", {}).get("atBats",  0) or 0) for s in last3)
        fb_pct = round(fly_balls / total_batted * 100, 1) if total_batted > 0 else None

        rows.append({"Player": name, "FB% (L3G)": fb_pct})

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── Last 3 games EV from Savant Statcast search ───────────────────────────────
@st.cache_data(ttl=1800)
def fetch_last3_ev_savant(team_id: int, season: int) -> pd.DataFrame:
    """
    Gets last 3 completed game dates for a team, then pulls individual
    Statcast batted ball events from Savant and averages EV per player.
    """
    sched_url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={team_id}&season={season}"
        f"&gameType=R&fields=dates,date,games,status,abstractGameState"
    )
    try:
        r = requests.get(sched_url, timeout=10)
        r.raise_for_status()
        dates_data = r.json().get("dates", [])
    except Exception:
        return pd.DataFrame()

    played = sorted(
        [d["date"] for d in dates_data
         if any(g.get("status", {}).get("abstractGameState") == "Final"
                for g in d.get("games", []))],
        reverse=True
    )[:3]

    if not played:
        return pd.DataFrame()

    start_date = played[-1]
    end_date   = played[0]

    url = (
        f"https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfGT=R%7C&hfSea={season}%7C&player_type=batter"
        f"&hfAB=single%7Cdouble%7Ctriple%7Chome_run%7Cfield_out%7Cgrounded_into_double_play"
        f"%7Cforce_out%7Cfield_error%7Csac_fly%7Csac_bunt%7Cdouble_play%7Ctriple_play%7C"
        f"&game_date_gt={start_date}&game_date_lt={end_date}"
        f"&team={team_id}&min_results=0&type=details&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if df.empty or "launch_speed" not in df.columns:
        return pd.DataFrame()

    name_col = next(
        (c for c in df.columns if c.lower() in ("player_name", "batter_name", "batter")),
        None
    )
    if not name_col:
        return pd.DataFrame()

    df["Player"] = df[name_col].apply(
        lambda x: " ".join(reversed([p.strip() for p in str(x).split(",")])) if "," in str(x) else str(x)
    )
    df["launch_speed"] = pd.to_numeric(df["launch_speed"], errors="coerce")
    df = df[df["launch_speed"] > 0]  # drop nulls and any zero readings

    return (
        df.groupby("Player")["launch_speed"]
        .mean().round(1).reset_index()
        .rename(columns={"launch_speed": "Avg EV (L3G)"})
    )

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
            "Pitcher":      player.get("fullName", "Unknown"),
            "Pitcher hand": player.get("pitchHand", {}).get("code", "R"),
            "HR/9":         round(hr_allowed / ip * 9, 2) if ip > 0 else 0.0,
        })
    return pd.DataFrame(rows)

# ── Fetch pitcher pitch mix from Savant ───────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_pitch_mix(season: int) -> pd.DataFrame:
    """
    Pulls pitch arsenal usage % for all pitchers from Savant for the given season.
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Name column
    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if name_col:
        df["Pitcher"] = df[name_col].apply(parse_savant_name)
    elif "player_name" in df.columns:
        df["Pitcher"] = df["player_name"].apply(parse_savant_name)
    else:
        return pd.DataFrame()

    pitch_col = next((c for c in df.columns if c.lower() in ("pitch_type", "pitch_name")), None)
    usage_col = next(
        (c for c in df.columns if "percent" in c.lower() and "pitch" in c.lower()),
        next((c for c in df.columns if c.lower() in ("pitch_usage", "pitch_percent", "pitches")), None)
    )

    if not pitch_col or not usage_col:
        return pd.DataFrame()

    df[usage_col] = pd.to_numeric(df[usage_col], errors="coerce")

    # If values look like raw counts (max >> 100), convert to % within each pitcher
    col_max = df[usage_col].dropna().max()

    rows = []
    for pitcher, group in df.groupby("Pitcher"):
        group = group[[pitch_col, usage_col]].dropna()
        if group.empty:
            continue

        total = group[usage_col].sum()
        if total == 0:
            continue

        # If raw counts, calculate % ourselves; if already %, use directly
        if col_max > 100:
            group = group.copy()
            group[usage_col] = (group[usage_col] / total * 100).round(1)
        elif col_max <= 1.0:
            group = group.copy()
            group[usage_col] = (group[usage_col] * 100).round(1)

        top = group.sort_values(usage_col, ascending=False).head(4)
        mix = ", ".join(
            f"{row[pitch_col]} {row[usage_col]:.0f}%"
            for _, row in top.iterrows()
            if row[usage_col] >= 5
        )
        if mix:
            rows.append({"Pitcher": pitcher, "Pitch mix": mix})

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── Fetch batter handedness splits (vs L / vs R) ──────────────────────────────
@st.cache_data(ttl=3600)
def fetch_batter_splits(season: int) -> pd.DataFrame:
    """
    Pulls each batter's SLG and HR split by opposing pitcher hand (L/R).
    One API call — returns all batters with vsLeft and vsRight splits.
    """
    rows = []
    for split_code, label in [("vl", "vs L"), ("vr", "vs R")]:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=statSplits&group=hitting&gameType=R"
            f"&season={season}&sitCodes={split_code}&limit=500"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            splits = r.json()["stats"][0]["splits"]
        except Exception:
            continue

        for s in splits:
            stat   = s.get("stat", {})
            player = s.get("player", {})
            ab     = int(stat.get("atBats", 0) or 0)
            hr     = int(stat.get("homeRuns", 0) or 0)
            tb     = int(stat.get("totalBases", 0) or 0)
            slg    = round(tb / ab, 3) if ab > 0 else None
            rows.append({
                "Player":         player.get("fullName", "Unknown"),
                "split":          label,
                f"HR ({label})":  hr,
                f"SLG ({label})": slg,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Pivot so each player has one row with both splits
    vl = df[df["split"] == "vs L"][["Player", "HR (vs L)", "SLG (vs L)"]].drop_duplicates("Player")
    vr = df[df["split"] == "vs R"][["Player", "HR (vs R)", "SLG (vs R)"]].drop_duplicates("Player")
    return vl.merge(vr, on="Player", how="outer")

# ── Fetch batter stats vs pitch type from Savant ──────────────────────────────
@st.cache_data(ttl=3600)
def fetch_pitch_type_splits(season: int) -> pd.DataFrame:
    """
    Pulls batter Barrel%, HH%, SwStr%, and Pull Air% broken down by pitch type.
    Savant pitch arsenal batter stats endpoint — one row per player per pitch type.
    Returns wide df: Player, then cols like Barrel%(FF), HH%(FF), SwStr%(FF) etc.
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=batter&pitchType=&year={season}&team=&min=10&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    name_col = next((c for c in df.columns if "last_name" in c.lower()), None)
    if name_col:
        df["Player"] = df[name_col].apply(parse_savant_name)
    else:
        return pd.DataFrame()

    # Always use pitch_type (codes like FF, SI, SL) not pitch_name for pivot keys
    # so they match the extract_top_pitch codes used in build_raw_rows
    pitch_col = next((c for c in df.columns if c.lower() == "pitch_type"), None)
    if not pitch_col:
        pitch_col = next((c for c in df.columns if c.lower() == "pitch_name"), None)
    if not pitch_col:
        return pd.DataFrame()

    # Confirmed columns from this endpoint:
    # whiff_percent, hard_hit_percent, woba, est_woba (xwOBA), slg, k_percent
    stat_map = {
        "hard_hit_percent": "HH%",
        "woba":             "wOBA",
        "est_woba":         "xwOBA",
        "k_percent":        "K%",
    }
    for col in stat_map:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pivot: one row per player, cols named Stat(PitchType)
    rows = {}
    for _, row in df.iterrows():
        player = row["Player"]
        pitch  = str(row[pitch_col]).strip()
        if player not in rows:
            rows[player] = {"Player": player}
        for src_col, display_name in stat_map.items():
            if src_col in df.columns:
                rows[player][f"{display_name} ({pitch})"] = row[src_col]

    return pd.DataFrame(list(rows.values()))

# ── 0-100 matchup score ────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "Avg EV (L3G)": 0.80,
    "FB% (L3G)":    0.20,
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
STATCAST_COLS = ["Avg EV", "Barrel%", "HH%"]
def build_raw_rows(batting_id, batting_team, opp_pitcher, home_team,
                   hitting_df, pitcher_df, min_ab, min_hr):
    batters = hitting_df[
        (hitting_df["team_id"] == batting_id) &
        (hitting_df["AB"] >= min_ab) &
        (hitting_df["HR"] >= min_hr)
    ]
    if batters.empty:
        return []

    pr          = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher]
    opp_hand    = pr.iloc[0]["Pitcher hand"] if not pr.empty else "R"
    split_label = f"vs {opp_hand}"

    # Get pitcher's top pitch type for pitch-split columns
    top_pitch = pr.iloc[0].get("Top pitch", None) if not pr.empty else None

    rows = []
    for _, b in batters.iterrows():
        row = {
            "Player":               b["Player"],
            "Batting team":         batting_team,
            "Opp pitcher":          opp_pitcher,
            "HR":                   b["HR"],
            "AB":                   b["AB"],
            "SLG":                  b["SLG"],
            f"HR ({split_label})":  b.get(f"HR ({split_label})", None),
            f"SLG ({split_label})": b.get(f"SLG ({split_label})", None),
            "Avg EV (L3G)":         b.get("Avg EV (L3G)", None),
            "FB% (L3G)":            b.get("FB% (L3G)", None),
            "Matchup score":        None,
            "_split_label":         split_label,
            "_top_pitch":           top_pitch,
        }
        # Season Statcast stats
        for col in STATCAST_COLS:
            row[col] = b.get(col, None)
        # Pitch-type split stats vs top pitch
        if top_pitch:
            for stat in ["HH%", "wOBA", "xwOBA", "K%"]:
                src = f"{stat} ({top_pitch})"
                row[f"{stat} vs top pitch"] = b.get(src, None)
        rows.append(row)
    return rows

# ── Display columns ────────────────────────────────────────────────────────────
BASE_DISPLAY_COLS = [
    "Player", "Batting team", "Opp pitcher",
    "HR", "AB",
    "HR (vs R)", "SLG (vs R)",
    "HR (vs L)", "SLG (vs L)",
    "Avg EV (L3G)", "FB% (L3G)",
    "Barrel%", "HH%",
    "HH% vs top pitch", "wOBA vs top pitch",
    "xwOBA vs top pitch", "K% vs top pitch",
    "SLG",
    "Matchup score",
]

HIGH_GOOD = [
    "Avg EV (L3G)", "FB% (L3G)", "Barrel%", "HH%",
    "HH% vs top pitch", "wOBA vs top pitch", "xwOBA vs top pitch",
    "SLG", "SLG (vs R)", "SLG (vs L)", "Matchup score",
]
LOW_GOOD = ["K% vs top pitch"]

def style_table(df: pd.DataFrame, cols: list):
    fmt = {
        "Avg EV (L3G)":        "{:.1f}",
        "FB% (L3G)":           "{:.1f}%",
        "Barrel%":             "{:.1f}%",
        "HH%":                 "{:.1f}%",
        "HH% vs top pitch":    "{:.1f}%",
        "wOBA vs top pitch":   "{:.3f}",
        "xwOBA vs top pitch":  "{:.3f}",
        "K% vs top pitch":     "{:.1f}%",
        "SLG":                 "{:.3f}",
        "SLG (vs R)":          "{:.3f}",
        "SLG (vs L)":          "{:.3f}",
        "Matchup score":       "{:.1f}",
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

    st.markdown("---")
    st.markdown("**Minimum filters**")
    min_ab = st.slider("Min at bats", 0, 100, 5, step=5)
    min_hr = st.slider("Min HR",      0,  20,  0, step=1)

    st.markdown("---")
    st.markdown("**Colour guide**")
    st.markdown("🟢 Green = best · 🟡 Yellow = average · 🔴 Red = bad")
    st.markdown("---")
    st.markdown("**Matchup score (0–100)**")
    st.caption("100 = best matchup on today's slate.")
    st.caption("• Avg EV (last 3 games) — 80%")
    st.caption("• FB% (last 3 games) — 20%")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.caption("MLB Stats API — HR, SLG, schedule, pitchers, splits")
    st.caption("Baseball Savant — EV, HH%, Barrel%, pitch mix, vs pitch type")

# ── Load base data ─────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh hourly · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule, Statcast, and pitcher data..."):
    games             = fetch_schedule(date_str)
    sc_main           = fetch_savant_main(season)
    sc_barrels        = fetch_savant_barrels(season)
    sc_fb             = fetch_savant_fb(season)
    pitcher_df        = fetch_pitcher_stats(season)
    splits_df         = fetch_batter_splits(season)
    pitch_mix_df      = fetch_pitch_mix(season)
    pitch_type_splits = fetch_pitch_type_splits(season)

# Add top pitch to pitcher_df from pitch_mix_df
if not pitch_mix_df.empty and "Pitch mix" in pitch_mix_df.columns:
    def extract_top_pitch(mix_str):
        if not mix_str or mix_str == "Pitch mix unavailable":
            return None
        first = mix_str.split(",")[0].strip()
        # Extract pitch type abbreviation from "Four-Seam Fastball 58%"
        # Map common names to Savant pitch type codes
        name_map = {
            "four-seam": "FF", "fastball": "FF", "sinker": "SI",
            "cutter": "FC", "slider": "SL", "curveball": "CU",
            "changeup": "CH", "splitter": "FS", "sweeper": "ST",
            "knuckleball": "KN", "screwball": "SC",
        }
        first_lower = first.lower()
        for key, code in name_map.items():
            if key in first_lower:
                return code
        return None
    pitch_mix_df["Top pitch"] = pitch_mix_df["Pitch mix"].apply(extract_top_pitch)
    pitcher_df = pitcher_df.merge(
        pitch_mix_df[["Pitcher", "Pitch mix", "Top pitch"]],
        on="Pitcher", how="left"
    )

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

if not splits_df.empty:
    hitting_df = hitting_df.merge(splits_df, on="Player", how="left")

if not pitch_type_splits.empty:
    st.write("Barrel% sample:", hitting_df["Barrel%"].dropna().head(3).tolist() if "Barrel%" in hitting_df.columns else "MISSING")
st.write("Pitch splits cols sample:", [c for c in hitting_df.columns if "vs top" in c][:5])
st.write("Dylan Cease top pitch:", pitcher_df[pitcher_df["Pitcher"] == "Dylan Cease"]["Top pitch"].tolist() if "Top pitch" in pitcher_df.columns else "NO TOP PITCH COL")
st.write("Pitch type splits shape:", pitch_type_splits.shape if not pitch_type_splits.empty else "EMPTY")
hitting_df = hitting_df.merge(pitch_type_splits, on="Player", how="left")

for col in STATCAST_COLS:
    if col not in hitting_df.columns:
        hitting_df[col] = None

# ── Fetch and merge last 3 games EV/LA/FB% per team ──────────────────────────
team_ids = set()
for g in games:
    if g["away_id"]: team_ids.add((g["away_id"], g["away_team"]))
    if g["home_id"]: team_ids.add((g["home_id"], g["home_team"]))

with st.spinner("Loading last 3 games fly ball and exit velocity data..."):
    l3g_fb_frames = []
    l3g_ev_frames = []
    for tid, _ in team_ids:
        df_fb = fetch_last3_fb(tid, season)
        if not df_fb.empty:
            l3g_fb_frames.append(df_fb)
        df_ev = fetch_last3_ev_savant(tid, season)
        if not df_ev.empty:
            l3g_ev_frames.append(df_ev)

# Merge FB% (L3G) — by Player name
if l3g_fb_frames:
    l3g_fb = (
        pd.concat(l3g_fb_frames)
        .drop_duplicates(subset=["Player"])
        .reset_index(drop=True)
    )
    hitting_df = hitting_df.merge(l3g_fb[["Player", "FB% (L3G)"]], on="Player", how="left")
else:
    hitting_df["FB% (L3G)"] = None

# Merge Avg EV (L3G) — by Player name from Savant
if l3g_ev_frames:
    l3g_ev = pd.concat(l3g_ev_frames).groupby("Player")["Avg EV (L3G)"].mean().round(1).reset_index()
    hitting_df = hitting_df.merge(l3g_ev, on="Player", how="left")
else:
    hitting_df["Avg EV (L3G)"] = None



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
        # Look up pitch mix for this pitcher
        pm_row  = pitch_mix_df[pitch_mix_df["Pitcher"] == opp_pitcher] if not pitch_mix_df.empty else pd.DataFrame()
        mix_str = pm_row.iloc[0]["Pitch mix"] if not pm_row.empty else "Pitch mix unavailable"

        st.markdown(
            f"<div style='background:var(--color-background-secondary);"
            f"border-left:3px solid {border};"
            f"padding:10px 14px;border-radius:var(--border-radius-md);margin:10px 0 4px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px'>"
            f"<span style='font-size:14px;font-weight:500'>{batting_team}</span>"
            f"<span style='font-size:12px;color:var(--color-text-secondary)'>batting vs {opp_pitcher}</span>"
            f"</div>"
            f"<div style='font-size:11px;color:var(--color-text-tertiary);margin-top:4px'>"
            f"🎯 Pitch mix: {mix_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        team_df = (
            full_df[full_df["_key"] == key]
            .drop(columns=["_key", "_split_label", "_top_pitch"], errors="ignore")
            .sort_values("Matchup score", ascending=False)
            .reset_index(drop=True)
        )
        team_df.index += 1

        if team_df.empty:
            st.info(f"No {batting_team} batters match current filters.")
        else:
            # Show only the relevant split columns (vs R or vs L)
            present_split = next(
                (c.split("HR (")[1].rstrip(")") for c in team_df.columns
                 if c.startswith("HR (vs")), None
            )
            dynamic_cols = [
                c for c in BASE_DISPLAY_COLS if c in team_df.columns
                and not (present_split and c in (
                    f"HR (vs {'L' if present_split == 'R' else 'R'})",
                    f"SLG (vs {'L' if present_split == 'R' else 'R'})"
                ))
            ]
            st.dataframe(style_table(team_df, dynamic_cols), use_container_width=True, height=380)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Top picks chart ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top picks across today's slate")

chart_df = (
    full_df.drop(columns=["_key", "_split_label", "_top_pitch"], errors="ignore")
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
    st.markdown("**Avg EV vs HH% — last 3 games**")
    plot_df = chart_df.head(60)
    hh_ok = "HH%"          in plot_df.columns and plot_df["HH%"].notna().any()
    ev_ok = "Avg EV (L3G)" in plot_df.columns and plot_df["Avg EV (L3G)"].notna().any()
    if hh_ok and ev_ok:
        fig2 = px.scatter(
            plot_df, x="HH%", y="Avg EV (L3G)", text="Player",
            color="Matchup score", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            size="HR",
            hover_data=["Batting team", "Opp pitcher", "SLG", "FB% (L3G)"],
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
