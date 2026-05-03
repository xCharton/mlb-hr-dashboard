import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, date

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

# ── Fetch today's schedule ─────────────────────────────────────────────────────
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
                "away_team":    away_team,
                "home_team":    home_team,
                "away_id":      away_id,
                "home_id":      home_id,
                "away_pitcher": away_pitcher,
                "home_pitcher": home_pitcher,
                "venue":        venue,
                "time":         time_str,
                "label":        f"{away_team} @ {home_team}  —  {time_str}",
            })
    return games

# ── Fetch season hitting stats ─────────────────────────────────────────────────
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
        ab  = int(stat.get("atBats", 1) or 1)
        rbi = int(stat.get("rbi", 0) or 0)
        bb  = int(stat.get("baseOnBalls", 0) or 0)
        so  = int(stat.get("strikeOuts", 0) or 0)

        def sf(key):
            v = stat.get(key)
            try:
                return float(v) if v not in (None, ".---", "-.--") else 0.0
            except (ValueError, TypeError):
                return 0.0

        avg = sf("avg")
        slg = sf("sluggingPercentage")
        obp = sf("obp")
        ops = sf("ops")
        iso = round(slg - avg, 3)

        rows.append({
            "player_id": player.get("id"),
            "Player":    player.get("fullName", "Unknown"),
            "team_id":   team.get("id"),
            "Team":      team.get("name", "Unknown"),
            "PA": pa, "AB": ab, "HR": hr,
            "HR/PA":  round(hr / pa, 4) if pa > 0 else 0.0,
            "AB/HR":  round(ab / hr, 1) if hr > 0 else 0.0,
            "AVG": avg, "OBP": obp, "SLG": slg, "OPS": ops, "ISO": iso,
            "RBI": rbi,
            "BB%": round(bb / pa * 100, 1) if pa > 0 else 0.0,
            "K%":  round(so / pa * 100, 1) if pa > 0 else 0.0,
        })
    return pd.DataFrame(rows)

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

        def sf(key):
            v = stat.get(key)
            try:
                return float(v) if v not in (None, ".---", "-.--") else 0.0
            except (ValueError, TypeError):
                return 0.0

        rows.append({
            "Pitcher":    player.get("fullName", "Unknown"),
            "IP":         ip,
            "HR allowed": hr_allowed,
            "HR/9":       round(hr_allowed / ip * 9, 2) if ip > 0 else 0.0,
            "ERA":        sf("era"),
            "WHIP":       sf("whip"),
            "K/9":        sf("strikeoutsPer9Inn"),
        })
    return pd.DataFrame(rows)

# ── Build matchup table for one game ──────────────────────────────────────────
def build_matchup_df(game, hitting_df, pitcher_df, min_pa, min_hr):
    rows = []
    matchups = [
        (game["away_id"], game["away_team"], game["home_pitcher"], game["home_team"]),
        (game["home_id"], game["home_team"], game["away_pitcher"], game["away_team"]),
    ]
    for batting_id, batting_team, opp_pitcher_name, opp_team in matchups:
        batters = hitting_df[
            (hitting_df["team_id"] == batting_id) &
            (hitting_df["PA"] >= min_pa) &
            (hitting_df["HR"] >= min_hr)
        ]
        if batters.empty:
            continue

        park_factor = PARK_FACTORS.get(game["home_team"], 1.00)

        pr = pitcher_df[pitcher_df["Pitcher"] == opp_pitcher_name]
        if not pr.empty:
            opp_hr9  = pr.iloc[0]["HR/9"]
            opp_era  = pr.iloc[0]["ERA"]
            opp_whip = pr.iloc[0]["WHIP"]
        else:
            opp_hr9 = opp_era = opp_whip = None

        pitcher_mult = 1.0 + (opp_hr9 - 1.2) * 0.15 if opp_hr9 is not None else 1.0

        for _, b in batters.iterrows():
            score = round(
                b["HR/PA"] * max(b["ISO"], 0.001) * park_factor * pitcher_mult * 10000, 1
            )
            rows.append({
                "Player":         b["Player"],
                "Batting team":   batting_team,
                "Opp pitcher":    opp_pitcher_name,
                "Opp team":       opp_team,
                "HR":             b["HR"],
                "HR/PA":          b["HR/PA"],
                "ISO":            b["ISO"],
                "OPS":            b["OPS"],
                "AVG":            b["AVG"],
                "BB%":            b["BB%"],
                "K%":             b["K%"],
                "Park factor":    park_factor,
                "Pitcher HR/9":   opp_hr9,
                "Pitcher ERA":    opp_era,
                "Pitcher WHIP":   opp_whip,
                "Matchup score":  score,
            })

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
    min_hr        = st.slider("Min HR this season", 0, 20, 3, step=1)

    st.markdown("---")
    st.markdown("**Columns to show**")
    show_iso    = st.checkbox("ISO",           value=True)
    show_ops    = st.checkbox("OPS",           value=True)
    show_avg    = st.checkbox("AVG",           value=False)
    show_bbk    = st.checkbox("BB% / K%",      value=False)
    show_pf     = st.checkbox("Park factor",   value=True)
    show_hr9    = st.checkbox("Pitcher HR/9",  value=True)
    show_era    = st.checkbox("Pitcher ERA",   value=True)
    show_whip   = st.checkbox("Pitcher WHIP",  value=False)

    st.markdown("---")
    st.markdown("**Matchup score**")
    st.caption("HR/PA × ISO × Park factor × Pitcher multiplier × 10,000")
    st.caption("Pitcher multiplier increases when facing a HR-prone starter.")

# ── Load data ──────────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
st.caption(
    f"Games for {selected_date.strftime('%A, %B %d, %Y')} · "
    f"Stats refresh every hour · Schedule refreshes every 30 min"
)

with st.spinner("Loading schedule and stats..."):
    games      = fetch_schedule(date_str)
    hitting_df = fetch_hitting_stats(season)
    pitcher_df = fetch_pitcher_stats(season)

if not games:
    st.warning(f"No games found for {date_str}. Try selecting a different date.")
    st.stop()

if hitting_df.empty:
    st.warning("Could not load hitting stats. Try again in a moment.")
    st.stop()

# ── Game cards ─────────────────────────────────────────────────────────────────
st.subheader("Today's games")
cols = st.columns(min(len(games), 3))
for i, g in enumerate(games):
    pf    = PARK_FACTORS.get(g["home_team"], 1.00)
    emoji = "🟢" if pf >= 1.05 else "🔴" if pf <= 0.95 else "⚪"
    label = "Hitter friendly" if pf >= 1.05 else "Pitcher friendly" if pf <= 0.95 else "Neutral"
    with cols[i % 3]:
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
    frames = [
        build_matchup_df(g, hitting_df, pitcher_df, min_pa, min_hr)
        for g in games
    ]
    frames = [f for f in frames if not f.empty]
    matchup_df = (
        pd.concat(frames).sort_values("Matchup score", ascending=False).reset_index(drop=True)
        if frames else pd.DataFrame()
    )
    if not matchup_df.empty:
        matchup_df.index += 1
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
c4.metric("At", top["Batting team"], f"PF {top['Park factor']:.2f}")

# ── Search ─────────────────────────────────────────────────────────────────────
search  = st.text_input("Search player or pitcher", placeholder="e.g. Judge, Cole")
view_df = matchup_df.copy()
if search:
    mask = (
        view_df["Player"].str.contains(search, case=False, na=False) |
        view_df["Opp pitcher"].str.contains(search, case=False, na=False)
    )
    view_df = view_df[mask]

# ── Build display columns ──────────────────────────────────────────────────────
base  = ["Player", "Batting team", "Opp pitcher", "HR", "HR/PA", "Matchup score"]
extra = []
if show_iso:  extra.append("ISO")
if show_ops:  extra.append("OPS")
if show_avg:  extra.append("AVG")
if show_bbk:  extra += ["BB%", "K%"]
if show_pf:   extra.append("Park factor")
if show_hr9:  extra.append("Pitcher HR/9")
if show_era:  extra.append("Pitcher ERA")
if show_whip: extra.append("Pitcher WHIP")

display_cols = [c for c in base + extra if c in view_df.columns]

fmt = {
    "HR/PA": "{:.4f}", "ISO": "{:.3f}", "OPS": "{:.3f}", "AVG": "{:.3f}",
    "BB%": "{:.1f}%", "K%": "{:.1f}%", "Park factor": "{:.2f}",
    "Pitcher HR/9": "{:.2f}", "Pitcher ERA": "{:.2f}", "Pitcher WHIP": "{:.2f}",
}
fmt = {k: v for k, v in fmt.items() if k in display_cols}

styled = view_df[display_cols].style.format(fmt, na_rep="TBD")
styled = styled.background_gradient(subset=["Matchup score"], cmap="YlOrRd")
styled = styled.background_gradient(subset=["HR/PA"],         cmap="Greens")
if "ISO"         in display_cols: styled = styled.background_gradient(subset=["ISO"],         cmap="Purples")
if "OPS"         in display_cols: styled = styled.background_gradient(subset=["OPS"],         cmap="Oranges")
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
        hover_data=["Batting team", "Opp pitcher", "HR", "HR/PA"],
        text="Opp pitcher",
    )
    fig.update_traces(textposition="inside", textfont_size=9)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=420,
        yaxis_title="", coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**HR/PA vs ISO — today's batters**")
    plot_df = matchup_df.head(60)
    fig2 = px.scatter(
        plot_df, x="ISO", y="HR/PA", text="Player",
        color="Matchup score", color_continuous_scale="YlOrRd",
        size="HR", hover_data=["Batting team", "Opp pitcher", "Matchup score"],
    )
    fig2.update_traces(textposition="top center", textfont_size=9)
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: MLB Stats API · Park factors are multi-year estimates · "
    "Probable pitchers from MLB schedule API · Bet responsibly."
)
