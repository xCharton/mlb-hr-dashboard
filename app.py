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

# ── Savant helpers ─────────────────────────────────────────────────────────────
def savant_name(row):
    if "last_name" in row.index and "first_name" in row.index:
        return f"{row['first_name'].strip()} {row['last_name'].strip()}"
    if "player_name" in row.index:
        name = str(row["player_name"])
        if "," in name:
            parts = [p.strip() for p in name.split(",")]
            return f"{parts[1]} {parts[0]}"
        return name
    return ""


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
            games.append({
                "away_team":    away.get("team", {}).get("name", "Unknown"),
                "home_team":    home.get("team", {}).get("name", "Unknown"),
                "away_id":      away.get("team", {}).get("id"),
                "home_id":      home.get("team", {}).get("id"),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "venue":        g.get("venue", {}).get("name", ""),
                "time":         _parse_time(g.get("gameDate", "")),
            })
    return games

def _parse_time(game_time):
    try:
        dt = datetime.strptime(game_time, "%Y-%m-%dT%H:%MZ")
        return dt.strftime("%-I:%M %p") + " ET"
    except Exception:
        return "TBD"


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


# ── Fetch Statcast (all metrics in one call) ───────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_statcast(season: int) -> pd.DataFrame:
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

    col_map = {
        "hard_hit_percent":   "HH%",
        "barrel_batted_rate": "Barrel%",
        "launch_speed":       "Avg EV",
        "launch_angle":       "Avg LA",
        "fb_percent":         "FB%",
        "pull_percent":       "Pull%",
    }
    rename = {}
    for src, dst in col_map.items():
        match = next((c for c in df.columns if src.lower() in c.lower()), None)
        if match:
            rename[match] = dst

    df = df.rename(columns=rename)
    keep = ["Player"] + [c for c in col_map.values() if c in df.columns]
    df   = df[keep].copy()
    for col in keep[1:]:
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


# ── Build matchup rows for one game, one batting team ─────────────────────────
STATCAST_COLS = ["HH%", "Barrel%", "Avg EV", "Avg LA", "FB%", "Pull%"]

def build_team_matchup(batting_id, batting_team, opp_pitcher_name, home_team,
                       hitting_df, pitcher_df, min_pa, min_hr):
    batters = hitting_df[
        (hitting_df["team_id"] == batting_id) &
        (hitting_df["PA"] >= min_pa) &
        (hitting_df["HR"] >= min_hr)
    ]
    if batters.empty:
        return pd.DataFrame()

    park_factor  = PARK_FACTORS.get(home_team, 1.00)
    pr           = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher_name]
    opp_hr9      = pr.iloc[0]["HR/9"] if not pr.empty else None
    pitcher_mult = 1.0 + (opp_hr9 - 1.2) * 0.15 if opp_hr9 is not None else 1.0

    rows = []
    for _, b in batters.iterrows():
        hh  = b.get("HH%",     40.0) or 40.0
        brl = b.get("Barrel%",  6.0) or 6.0
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
        for col in STATCAST_COLS:
            row[col] = b.get(col, None)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Matchup score", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ── Color coding: Green=best → Yellow → Orange → Red=worst ───────────────────
# Uses RdYlGn colormap (red=low, green=high) for stats where higher is better,
# and RdYlGn_r (reversed) for stats where lower is better.
def style_table(df: pd.DataFrame, display_cols: list) -> pd.io.formats.style.Styler:
    # Higher = better (green)
    high_good = {
        "HH%":          "RdYlGn",
        "Barrel%":      "RdYlGn",
        "Avg EV":       "RdYlGn",
        "FB%":          "RdYlGn",
        "Pull%":        "RdYlGn",
        "ISO":          "RdYlGn",
        "SLG":          "RdYlGn",
        "Matchup score":"RdYlGn",
    }
    # These stats: higher angle can be good but it's neutral — skip gradient
    # Pitcher HR/9: higher = worse for pitcher = better for batter prop
    pitcher_good = {
        "Pitcher HR/9": "RdYlGn",   # higher HR/9 = juicier matchup = green
    }
    # Park factor: higher = better for batters
    park_good = {
        "Park factor":  "RdYlGn",
    }

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
    fmt = {k: v for k, v in fmt.items() if k in display_cols}

    styled = df[display_cols].style.format(fmt, na_rep="—")

    for col, cmap in {**high_good, **pitcher_good, **park_good}.items():
        if col in display_cols and df[col].notna().any():
            styled = styled.background_gradient(subset=[col], cmap=cmap)

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
    st.markdown("🟢 Green = best · 🟡 Yellow = great")
    st.markdown("🟠 Orange = average · 🔴 Red = bad")
    st.markdown("---")
    st.markdown("**Matchup score**")
    st.caption("ISO × HH% × Barrel% × Park factor × Pitcher multiplier")
    st.caption("Higher = stronger HR prop candidate today.")


# ── Load data ──────────────────────────────────────────────────────────────────
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

# ── Merge Statcast ─────────────────────────────────────────────────────────────
if not statcast.empty:
    hitting_df = hitting_df.merge(statcast, on="Player", how="left")
    sc_loaded  = True
else:
    for col in STATCAST_COLS:
        hitting_df[col] = None
    sc_loaded = False

if not sc_loaded:
    st.warning(
        "⚠️ Baseball Savant data unavailable right now — Statcast columns show '—'. "
        "All other data is current."
    )

# ── Game selector ──────────────────────────────────────────────────────────────
st.subheader("Today's games")
all_label = f"All games today ({len(games)} games)"
options   = [all_label] + [
    f"{g['away_team']} @ {g['home_team']}  —  {g['time']}" for g in games
]
choice = st.selectbox("Filter by game", options)

selected_games = (
    games if choice == all_label
    else [g for g in games if f"{g['away_team']} @ {g['home_team']}  —  {g['time']}" == choice]
)

# ── Display columns ────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "Player", "Batting team", "Opp pitcher",
    "HR",
    "HH%", "Barrel%", "Avg EV", "Avg LA", "FB%", "Pull%",
    "ISO", "SLG",
    "Park factor", "Pitcher HR/9",
    "Matchup score",
]

# ── Render each game as a section ─────────────────────────────────────────────
any_data = False

for g in selected_games:
    away_team    = g["away_team"]
    home_team    = g["home_team"]
    away_pitcher = g["away_pitcher"]
    home_pitcher = g["home_pitcher"]
    venue        = g["venue"]
    time_str     = g["time"]
    pf           = PARK_FACTORS.get(home_team, 1.00)
    pf_emoji     = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    pf_label     = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral park"

    # Game header
    st.markdown("---")
    st.markdown(
        f"### {away_team} @ {home_team}",
    )
    st.caption(f"{time_str} · {venue} · {pf_emoji} {pf_label} (PF {pf:.2f})")

    # Two columns — one per batting team
    col_away, col_home = st.columns(2)

    # ── Away batters (face home pitcher) ──────────────────────────────────────
    with col_away:
        st.markdown(f"**{away_team}** batting vs {home_pitcher}")
        away_df = build_team_matchup(
            g["away_id"], away_team, home_pitcher, home_team,
            hitting_df, pitcher_df, min_pa, min_hr
        )
        if away_df.empty:
            st.info("No batters match current filters.")
        else:
            any_data = True
            valid = [c for c in DISPLAY_COLS if c in away_df.columns]
            st.dataframe(
                style_table(away_df, valid),
                use_container_width=True,
                height=420,
            )

    # ── Home batters (face away pitcher) ──────────────────────────────────────
    with col_home:
        st.markdown(f"**{home_team}** batting vs {away_pitcher}")
        home_df = build_team_matchup(
            g["home_id"], home_team, away_pitcher, home_team,
            hitting_df, pitcher_df, min_pa, min_hr
        )
        if home_df.empty:
            st.info("No batters match current filters.")
        else:
            any_data = True
            valid = [c for c in DISPLAY_COLS if c in home_df.columns]
            st.dataframe(
                style_table(home_df, valid),
                use_container_width=True,
                height=420,
            )

if not any_data:
    st.info("No batters match your filters across any game today. Try lowering Min PA or Min HR.")
    st.stop()

# ── Overall chart — top picks across all selected games ───────────────────────
st.markdown("---")
st.subheader("Top picks across today's slate")

all_frames = []
for g in selected_games:
    for batting_id, batting_team, opp_pitcher in [
        (g["away_id"], g["away_team"], g["home_pitcher"]),
        (g["home_id"], g["home_team"], g["away_pitcher"]),
    ]:
        df_t = build_team_matchup(
            batting_id, batting_team, opp_pitcher, g["home_team"],
            hitting_df, pitcher_df, min_pa, min_hr
        )
        if not df_t.empty:
            all_frames.append(df_t)

if all_frames:
    full_df = pd.concat(all_frames).sort_values("Matchup score", ascending=False).reset_index(drop=True)
    full_df.index += 1

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 15 matchup scores**")
        top15 = full_df.head(15).sort_values("Matchup score")
        fig = px.bar(
            top15, x="Matchup score", y="Player", orientation="h",
            color="Matchup score", color_continuous_scale="RdYlGn",
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
        plot_df = full_df.head(60)
        if "Barrel%" in plot_df.columns and plot_df["Barrel%"].notna().any():
            fig2 = px.scatter(
                plot_df, x="HH%", y="Barrel%", text="Player",
                color="Matchup score", color_continuous_scale="RdYlGn",
                size="HR",
                hover_data=["Batting team", "Opp pitcher", "ISO", "Avg EV", "FB%"],
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
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
