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

# ── Savant name helper ─────────────────────────────────────────────────────────
def savant_name(row):
    if "last_name" in row.index and "first_name" in row.index:
        return f"{str(row['first_name']).strip()} {str(row['last_name']).strip()}"
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
            "PA": pa, "HR": hr, "ISO": iso,
        })
    return pd.DataFrame(rows)

# ── Fetch Statcast — main leaderboard (HH%, Avg EV, Avg LA, FB%) ──────────────
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
    except Exception as e:
        st.warning(f"Savant fetch failed: {e}")
        return pd.DataFrame()

    st.info(f"Savant columns: {list(df.columns[:15])}")  # show us first 15 cols
    df["Player"] = df.apply(savant_name, axis=1)
    return df

# ── Fetch Statcast — bat tracking / swing metrics (SwStr%) ────────────────────
@st.cache_data(ttl=3600)
def fetch_statcast_swing(season: int) -> pd.DataFrame:
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd="
        f"&gameType=Regular+Season&isHardHit=&minSwings=100"
        f"&minGroupSwings=1&pitchType=&seasonStart={season}&seasonEnd={season}"
        f"&team=&type=batter&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    df["Player"] = df.apply(savant_name, axis=1)

    # SwStr% = whiff_percent in Savant bat-tracking
    swstr_col = next(
        (c for c in df.columns if "whiff" in c.lower()),
        next((c for c in df.columns if "swstr" in c.lower()), None)
    )
    if swstr_col is None:
        return pd.DataFrame()

    df = df.rename(columns={swstr_col: "SwStr%"})
    df["SwStr%"] = pd.to_numeric(df["SwStr%"], errors="coerce")
    return df[["Player", "SwStr%"]].dropna(subset=["Player"])

# ── Fetch Statcast — barrels (PulledBrl%, Brl/BIP%) ──────────────────────────
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

    df["Player"] = df.apply(savant_name, axis=1)

    col_map = {
        "barrel_batted_rate": "Brl/BIP%",
        "pulled_barrel":      "PulledBrl%",
        "brl_percent":        "Brl/BIP%",   # fallback name
    }
    rename = {}
    for src, dst in col_map.items():
        match = next((c for c in df.columns if src.lower() in c.lower()), None)
        if match and dst not in rename.values():
            rename[match] = dst

    df = df.rename(columns=rename)
    keep = ["Player"] + [c for c in ["Brl/BIP%", "PulledBrl%"] if c in df.columns]
    if len(keep) == 1:
        return pd.DataFrame()

    df = df[keep].copy()
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

# ── Build matchup for one batting team ────────────────────────────────────────
STATCAST_COLS = ["HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%", "SwStr%", "Brl/BIP%", "PulledBrl%"]

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
        brl = b.get("Brl/BIP%", 6.0) or 6.0
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

# ── Display columns (final order) ─────────────────────────────────────────────
DISPLAY_COLS = [
    "Player", "Batting team", "Opp pitcher",
    "HR",
    "HH%", "Avg EV", "Avg LA", "FB%", "SweetSpot%",
    "SwStr%", "Brl/BIP%", "PulledBrl%",
    "ISO",
    "Park factor", "Pitcher HR/9",
    "Matchup score",
]

# ── Colour coding: RdYlGn = red(bad)→yellow→green(best) ──────────────────────
# SwStr% is LOWER = better for batter (fewer whiffs), so reversed
HIGH_GOOD  = ["HH%", "Avg EV", "FB%", "SweetSpot%", "Brl/BIP%", "PulledBrl%",
              "ISO", "Park factor", "Pitcher HR/9", "Matchup score"]
LOW_GOOD   = ["SwStr%"]   # lower whiff rate = better contact = green when low

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
    }
    fmt = {k: v for k, v in fmt.items() if k in cols}

    styled = df[cols].style.format(fmt, na_rep="—")

    for col in HIGH_GOOD:
        if col in cols and df[col].notna().any():
            styled = styled.background_gradient(subset=[col], cmap="RdYlGn")

    for col in LOW_GOOD:
        if col in cols and df[col].notna().any():
            # Reversed: low SwStr% = green (good contact), high = red (lots of whiffs)
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
    st.markdown("🟢 Green = best")
    st.markdown("🟡 Yellow = average")
    st.markdown("🔴 Red = bad")
    st.caption("SwStr% is reversed — green means low whiff rate (good contact).")
    st.markdown("---")
    if st.checkbox("Show debug info"):
        st.write("Statcast main columns:", list(sc_main.columns) if not sc_main.empty else "EMPTY")
        st.write("Swing columns:", list(sc_swing.columns) if not sc_swing.empty else "EMPTY")
        st.write("Barrel columns:", list(sc_barrels.columns) if not sc_barrels.empty else "EMPTY")
        st.write("Hitting df columns:", list(hitting_df.columns))
        st.write("HH% sample:", hitting_df["HH%"].dropna().head(3).tolist() if "HH%" in hitting_df.columns else "MISSING")
    st.markdown("**Matchup score**")
    st.caption("ISO × HH% × Brl/BIP% × Park factor × Pitcher multiplier")

# ── Load data ──────────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh hourly · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule, MLB stats, and Statcast data..."):
    games          = fetch_schedule(date_str)
    hitting_df     = fetch_hitting_stats(season)
    sc_main        = fetch_statcast_main(season)
    sc_swing       = fetch_statcast_swing(season)
    sc_barrels     = fetch_statcast_barrels(season)
    pitcher_df     = fetch_pitcher_stats(season)  

if not games:
    st.warning(f"No games found for {date_str}. Try a different date.")
    st.stop()
if hitting_df.empty:
    st.warning("Could not load hitting stats. Try again in a moment.")
    st.stop()

# ── Merge all Statcast data ────────────────────────────────────────────────────
for sc_df in [sc_main, sc_swing, sc_barrels]:
    if not sc_df.empty:
        hitting_df = hitting_df.merge(sc_df, on="Player", how="left")

# Fill any missing Statcast cols with None
for col in STATCAST_COLS:
    if col not in hitting_df.columns:
        hitting_df[col] = None

sc_loaded = not sc_main.empty
if not sc_loaded:
    st.warning(
        "⚠️ Baseball Savant data unavailable right now — "
        "Statcast columns show '—'. All other data is current."
    )

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

# ── Render each game — teams stacked vertically ────────────────────────────────
any_data = False

for g in selected_games:
    away_team    = g["away_team"]
    home_team    = g["home_team"]
    away_pitcher = g["away_pitcher"]
    home_pitcher = g["home_pitcher"]
    pf           = PARK_FACTORS.get(home_team, 1.00)
    pf_emoji     = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    pf_label     = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral park"

    st.markdown("---")

    # ── Game header ───────────────────────────────────────────────────────────
    st.markdown(f"### ⚾ {away_team} @ {home_team}")
    st.caption(f"{g['time']} · {g['venue']} · {pf_emoji} {pf_label} (PF {pf:.2f})")

    # ── AWAY batting team (stacked on top) ────────────────────────────────────
    st.markdown(
        f"<div style='background:var(--color-background-secondary);"
        f"border-left:3px solid var(--color-border-info);"
        f"padding:8px 14px;border-radius:var(--border-radius-md);margin:10px 0 4px'>"
        f"<span style='font-size:14px;font-weight:500'>{away_team}</span>"
        f"<span style='font-size:12px;color:var(--color-text-secondary)'>"
        f" batting vs {home_pitcher}</span></div>",
        unsafe_allow_html=True
    )
    away_df = build_team_matchup(
        g["away_id"], away_team, home_pitcher, home_team,
        hitting_df, pitcher_df, min_pa, min_hr
    )
    if away_df.empty:
        st.info(f"No {away_team} batters match current filters.")
    else:
        any_data = True
        valid    = [c for c in DISPLAY_COLS if c in away_df.columns]
        st.dataframe(style_table(away_df, valid), use_container_width=True, height=380)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── HOME batting team (stacked below) ────────────────────────────────────
    st.markdown(
        f"<div style='background:var(--color-background-secondary);"
        f"border-left:3px solid var(--color-border-success);"
        f"padding:8px 14px;border-radius:var(--border-radius-md);margin:4px 0 4px'>"
        f"<span style='font-size:14px;font-weight:500'>{home_team}</span>"
        f"<span style='font-size:12px;color:var(--color-text-secondary)'>"
        f" batting vs {away_pitcher}</span></div>",
        unsafe_allow_html=True
    )
    home_df = build_team_matchup(
        g["home_id"], home_team, away_pitcher, home_team,
        hitting_df, pitcher_df, min_pa, min_hr
    )
    if home_df.empty:
        st.info(f"No {home_team} batters match current filters.")
    else:
        any_data = True
        valid    = [c for c in DISPLAY_COLS if c in home_df.columns]
        st.dataframe(style_table(home_df, valid), use_container_width=True, height=380)

if not any_data:
    st.info("No batters match your filters. Try lowering Min PA or Min HR.")
    st.stop()

# ── Top picks chart across full slate ─────────────────────────────────────────
st.markdown("---")
st.subheader("Top picks across today's slate")

all_frames = []
for g in selected_games:
    for bid, bt, opp in [
        (g["away_id"], g["away_team"], g["home_pitcher"]),
        (g["home_id"], g["home_team"], g["away_pitcher"]),
    ]:
        df_t = build_team_matchup(bid, bt, opp, g["home_team"],
                                  hitting_df, pitcher_df, min_pa, min_hr)
        if not df_t.empty:
            all_frames.append(df_t)

if all_frames:
    full_df = (
        pd.concat(all_frames)
        .sort_values("Matchup score", ascending=False)
        .reset_index(drop=True)
    )
    full_df.index += 1

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 15 matchup scores**")
        top15 = full_df.head(15).sort_values("Matchup score")
        fig = px.bar(
            top15, x="Matchup score", y="Player", orientation="h",
            color="Matchup score", color_continuous_scale="RdYlGn",
            hover_data=["Batting team", "Opp pitcher", "HR", "HH%", "Brl/BIP%"],
            text="Opp pitcher",
        )
        fig.update_traces(textposition="inside", textfont_size=9)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=440,
            yaxis_title="", coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Brl/BIP% vs HH% — today's batters**")
        plot_df = full_df.head(60)
        hh_ok  = "HH%"      in plot_df.columns and plot_df["HH%"].notna().any()
        brl_ok = "Brl/BIP%" in plot_df.columns and plot_df["Brl/BIP%"].notna().any()
        if hh_ok and brl_ok:
            fig2 = px.scatter(
                plot_df, x="HH%", y="Brl/BIP%", text="Player",
                color="Matchup score", color_continuous_scale="RdYlGn",
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
    "Data: MLB Stats API + Baseball Savant Statcast · "
    "Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
