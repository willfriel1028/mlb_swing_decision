# Dash imports
import dash
from dash import dcc, html, Input, Output, dash_table

# Pandas & numpy
import pandas as pd
import numpy as np

# DuckDB
import duckdb

# Pybaseball imports
from pybaseball import statcast
from pybaseball import playerid_reverse_lookup
from pybaseball import batting_stats_range

# To attain current date
from datetime import date

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Creates the Dash application object
app = dash.Dash(__name__)

server = app.server

# Load in updated data from parquet
df = pd.read_parquet('season_data.parquet')

# This function returns the default view dataframe that the dashboard automatically presents
def default(count=None, outs=None, bases=None, phand=None, pitch_type=None):
    where_clause = build_where_clause(count, outs, bases, phand, pitch_type)

    query = f"""
        WITH agg AS (
            SELECT
                batter,
                FIRST(batter_name) AS player,
                FIRST(stand) AS bats,
                FIRST(batting_team) AS team,
                FIRST(PA) AS PA,
                SUM(correct_sd) AS correct_sd,
                SUM(strike_swing) AS strike_swing,
                SUM(strike_taken) AS strike_taken,
                SUM(ball_swing) AS ball_swing,
                SUM(ball_taken) AS ball_taken,
                COUNT(*) AS pitches,
                SUM(strike_swing) * 100.0 / (SUM(strike_swing) + SUM(strike_taken)) AS iz_raw,
                SUM(ball_swing) * 100.0 / (SUM(ball_taken) + SUM(ball_swing)) AS ooz_raw,
                SUM(correct_sd) * 100.0 / COUNT(*) AS sd_raw
            FROM 'season_data.parquet'
            WHERE {where_clause}
            GROUP BY batter
        )
        SELECT
            * EXCLUDE (iz_raw, ooz_raw, sd_raw),
            ROUND(iz_raw, 1) AS "z-swing%",
            ROUND(ooz_raw, 1) AS "o-swing%",
            ROUND(sd_raw, 1) AS "sd%",
            CASE WHEN pitches >= 150 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches >= 150 ORDER BY iz_raw) * 100) ELSE NULL END AS "z-swing",
            CASE WHEN pitches >= 150 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches >= 150 ORDER BY ooz_raw DESC) * 100) ELSE NULL END AS "o-swing",
            CASE WHEN pitches >= 150 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches >= 150 ORDER BY sd_raw) * 100) ELSE NULL END AS sd
        FROM agg
    """
    chart1 = duckdb.sql(query).df()
    chart1 = chart1[["batter", "player", "bats", "team", "PA", "pitches", "z-swing%", "z-swing", "o-swing%", "o-swing", "sd%", "sd"]]
    chart1 = chart1.sort_values("PA", ascending=False)
    return chart1

# This function returns the version of the dataframe that is split by the handedness of the opposing pitcher
def pitcher_hand(count=None, outs=None, bases=None, phand=None, pitch_type=None):
    where_clause = build_where_clause(count, outs, bases, phand, pitch_type)

    query = f"""
        SELECT
            batter,
            p_throws,
            FIRST(batter_name) AS player,
            FIRST(stand) AS bats,
            FIRST(batting_team) AS team,
            FIRST(PA) AS PA,
            SUM(correct_sd) AS correct_sd,
            SUM(strike_swing) AS strike_swing,
            SUM(strike_taken) AS strike_taken,
            SUM(ball_swing) AS ball_swing,
            SUM(ball_taken) AS ball_taken,
            COUNT(*) AS pitches
        FROM 'season_data.parquet'
        WHERE {where_clause}
        GROUP BY batter, p_throws
    """
    y = duckdb.sql(query).df()

    # Columns to attain for both right and left handed pitchers
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'pitches']

    # Pivot data
    chart2 = y.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns='p_throws',
        values=metric_cols
    )

    chart2.columns = [f'{col}_{side}' for col, side in chart2.columns]
    chart2 = chart2.reset_index()

    # Add any missing R/L columns as all-NaN (happens when phand filters out one hand entirely)
    metric_side_cols = [f"{m}_{s}" for m in metric_cols for s in ["R", "L"]]
    missing_cols = [c for c in metric_side_cols if c not in chart2.columns]
    for col in missing_cols:
        chart2[col] = np.nan

    chart2[metric_side_cols] = chart2[metric_side_cols].apply(pd.to_numeric, errors="coerce")

    team_lookup = y.groupby('batter')['team'].first().rename('team')
    chart2 = chart2.merge(team_lookup, on='batter', how='left')

    query2 = """
        SELECT
            * EXCLUDE (iz_raw_R, iz_raw_L, ooz_raw_R, ooz_raw_L, sd_raw_R, sd_raw_L),
            ROUND(iz_raw_R, 1) AS "z-swing% vs RHP",
            ROUND(iz_raw_L, 1) AS "z-swing% vs LHP",
            ROUND(ooz_raw_R, 1) AS "o-swing% vs RHP",
            ROUND(ooz_raw_L, 1) AS "o-swing% vs LHP",
            ROUND(sd_raw_R, 1) AS "sd% vs RHP",
            ROUND(sd_raw_L, 1) AS "sd% vs LHP",
            CASE WHEN pitches_R >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R >= 100 ORDER BY iz_raw_R) * 100) ELSE NULL END AS "z-swing vs RHP",
            CASE WHEN pitches_L >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L >= 100 ORDER BY iz_raw_L) * 100) ELSE NULL END AS "z-swing vs LHP",
            CASE WHEN pitches_R >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R >= 100 ORDER BY ooz_raw_R DESC) * 100) ELSE NULL END AS "o-swing vs RHP",
            CASE WHEN pitches_L >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L >= 100 ORDER BY ooz_raw_L DESC) * 100) ELSE NULL END AS "o-swing vs LHP",
            CASE WHEN pitches_R >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R >= 100 ORDER BY sd_raw_R) * 100) ELSE NULL END AS "sd vs RHP",
            CASE WHEN pitches_L >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L >= 100 ORDER BY sd_raw_L) * 100) ELSE NULL END AS "sd vs LHP"
        FROM (
            SELECT
                *,
                strike_swing_R * 100.0 / NULLIF(strike_swing_R + strike_taken_R, 0) AS iz_raw_R,
                strike_swing_L * 100.0 / NULLIF(strike_swing_L + strike_taken_L, 0) AS iz_raw_L,
                ball_swing_R * 100.0 / NULLIF(ball_taken_R + ball_swing_R, 0) AS ooz_raw_R,
                ball_swing_L * 100.0 / NULLIF(ball_taken_L + ball_swing_L, 0) AS ooz_raw_L,
                correct_sd_R * 100.0 / NULLIF(pitches_R, 0) AS sd_raw_R,
                correct_sd_L * 100.0 / NULLIF(pitches_L, 0) AS sd_raw_L
            FROM chart2
        )
    """
    chart2 = duckdb.sql(query2).df()

    chart2["pitches"] = round(chart2["pitches_R"].add(chart2["pitches_L"], fill_value=0))

    chart2 = chart2[['batter', 'player', 'bats', 'team', 'PA', 'pitches', 
                     'z-swing% vs RHP', 'o-swing% vs RHP', 'sd% vs RHP', 
                     'z-swing% vs LHP', 'o-swing% vs LHP', 'sd% vs LHP', 
                     'z-swing vs RHP', 'o-swing vs RHP', 'sd vs RHP', 
                     'z-swing vs LHP', 'o-swing vs LHP', 'sd vs LHP']]
    chart2 = chart2.sort_values("PA", ascending=False)
    return chart2

# This function returns the version of the dataframe that is split by the pitch type the batter is facing
def pitch_group(count=None, outs=None, bases=None, phand=None, pitch_type=None):
    where_clause = build_where_clause(count, outs, bases, phand, pitch_type)

    query = f"""
        SELECT
            batter,
            pitch_group,
            FIRST(batter_name) AS player,
            FIRST(stand) AS bats,
            FIRST(batting_team) AS team,
            FIRST(PA) AS PA,
            SUM(correct_sd) AS correct_sd,
            SUM(strike_swing) AS strike_swing,
            SUM(strike_taken) AS strike_taken,
            SUM(ball_swing) AS ball_swing,
            SUM(ball_taken) AS ball_taken,
            COUNT(*) AS pitches
        FROM 'season_data.parquet'
        WHERE {where_clause}
        GROUP BY batter, pitch_group
    """
    y = duckdb.sql(query).df()

    # Columns to attain for all pitch groups
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'pitches']

    # Pivot data
    chart3 = y.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns='pitch_group',
        values=metric_cols
    )

    chart3.columns = [f'{col}_{group}' for col, group in chart3.columns]
    chart3 = chart3.reset_index()

    # Add any missing pitch group columns as all-NaN (happens when pitch_type filters out others entirely)
    metric_side_cols = [f"{m}_{g}" for m in metric_cols for g in ["FA", "OFF", "BB"]]
    missing_cols = [c for c in metric_side_cols if c not in chart3.columns]
    for col in missing_cols:
        chart3[col] = np.nan

    chart3[metric_side_cols] = chart3[metric_side_cols].apply(pd.to_numeric, errors="coerce")

    team_lookup = y.groupby('batter')['team'].first().rename('team')
    chart3 = chart3.merge(team_lookup, on='batter', how='left')

    query2 = """
        SELECT
            * EXCLUDE (iz_raw_FA, iz_raw_OFF, iz_raw_BB, ooz_raw_FA, ooz_raw_OFF, ooz_raw_BB, sd_raw_FA, sd_raw_OFF, sd_raw_BB),
            ROUND(iz_raw_FA, 1) AS "z-swing% vs FA",
            ROUND(iz_raw_OFF, 1) AS "z-swing% vs OFF",
            ROUND(iz_raw_BB, 1) AS "z-swing% vs BB",
            ROUND(ooz_raw_FA, 1) AS "o-swing% vs FA",
            ROUND(ooz_raw_OFF, 1) AS "o-swing% vs OFF",
            ROUND(ooz_raw_BB, 1) AS "o-swing% vs BB",
            ROUND(sd_raw_FA, 1) AS "sd% vs FA",
            ROUND(sd_raw_OFF, 1) AS "sd% vs OFF",
            ROUND(sd_raw_BB, 1) AS "sd% vs BB",
            CASE WHEN pitches_FA >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_FA >= 100 ORDER BY iz_raw_FA) * 100) ELSE NULL END AS "z-swing vs FA",
            CASE WHEN pitches_FA >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_FA >= 100 ORDER BY ooz_raw_FA DESC) * 100) ELSE NULL END AS "o-swing vs FA",
            CASE WHEN pitches_FA >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_FA >= 100 ORDER BY sd_raw_FA) * 100) ELSE NULL END AS "sd vs FA",
            CASE WHEN pitches_OFF >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_OFF >= 100 ORDER BY iz_raw_OFF) * 100) ELSE NULL END AS "z-swing vs OFF",
            CASE WHEN pitches_OFF >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_OFF >= 100 ORDER BY ooz_raw_OFF DESC) * 100) ELSE NULL END AS "o-swing vs OFF",
            CASE WHEN pitches_OFF >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_OFF >= 100 ORDER BY sd_raw_OFF) * 100) ELSE NULL END AS "sd vs OFF",
            CASE WHEN pitches_BB >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_BB >= 100 ORDER BY iz_raw_BB) * 100) ELSE NULL END AS "z-swing vs BB",
            CASE WHEN pitches_BB >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_BB >= 100 ORDER BY ooz_raw_BB DESC) * 100) ELSE NULL END AS "o-swing vs BB",
            CASE WHEN pitches_BB >= 100 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_BB >= 100 ORDER BY sd_raw_BB) * 100) ELSE NULL END AS "sd vs BB",
        FROM (
            SELECT
                *,
                strike_swing_FA * 100.0 / NULLIF(strike_swing_FA + strike_taken_FA, 0) AS iz_raw_FA,
                strike_swing_OFF * 100.0 / NULLIF(strike_swing_OFF + strike_taken_OFF, 0) AS iz_raw_OFF,
                strike_swing_BB * 100.0 / NULLIF(strike_swing_BB + strike_taken_BB, 0) AS iz_raw_BB,
                ball_swing_FA * 100.0 / NULLIF(ball_swing_FA + ball_taken_FA, 0) AS ooz_raw_FA,
                ball_swing_OFF * 100.0 / NULLIF(ball_swing_OFF + ball_taken_OFF, 0) AS ooz_raw_OFF,
                ball_swing_BB * 100.0 / NULLIF(ball_swing_BB + ball_taken_BB, 0) AS ooz_raw_BB,
                correct_sd_FA * 100.0 / NULLIF(pitches_FA, 0) AS sd_raw_FA,
                correct_sd_OFF * 100.0 / NULLIF(pitches_OFF, 0) AS sd_raw_OFF,
                correct_sd_BB * 100.0 / NULLIF(pitches_BB, 0) AS sd_raw_BB,
            FROM chart3
        )
    """
    chart3 = duckdb.sql(query2).df()

    chart3["pitches"] = round(
        chart3["pitches_FA"]
        .add(chart3["pitches_OFF"], fill_value=0)
        .add(chart3["pitches_BB"], fill_value=0)
    )

    chart3 = chart3[['batter', 'player', 'bats', 'team', 'PA', 'pitches', 
                     'z-swing% vs FA', 'o-swing% vs FA', 'sd% vs FA', 
                     'z-swing% vs OFF', 'o-swing% vs OFF', 'sd% vs OFF', 
                     'z-swing% vs BB', 'o-swing% vs BB', 'sd% vs BB',
                     'z-swing vs FA', 'o-swing vs FA', 'sd vs FA', 
                     'z-swing vs OFF', 'o-swing vs OFF', 'sd vs OFF',
                     'z-swing vs BB', 'o-swing vs BB', 'sd vs BB']]
    chart3 = chart3.sort_values("PA", ascending=False)
    return chart3

# This function returns the version of the dataframe that is split by the opposing pitcher handedness and the pitch type the batter is facing
def hand_group(count=None, outs=None, bases=None, phand=None, pitch_type=None):
    where_clause = build_where_clause(count, outs, bases, phand, pitch_type)

    query = f"""
        SELECT
            batter,
            p_throws,
            pitch_group,
            FIRST(batter_name) AS player,
            FIRST(stand) AS bats,
            FIRST(batting_team) AS team,
            FIRST(PA) AS PA,
            SUM(correct_sd) AS correct_sd,
            SUM(strike_swing) AS strike_swing,
            SUM(strike_taken) AS strike_taken,
            SUM(ball_swing) AS ball_swing,
            SUM(ball_taken) AS ball_taken,
            COUNT(*) AS pitches
        FROM 'season_data.parquet'
        WHERE {where_clause}
        GROUP BY batter, p_throws, pitch_group
    """
    y = duckdb.sql(query).df()

    # Columns to attain for all pitcher handedness + pitch group combos
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'pitches']

    # Pivot data
    chart4 = y.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns=['p_throws', 'pitch_group'],
        values=metric_cols
    )

    # Create a column for each combo of RHP/LHP and pitch group with columns of interest
    chart4.columns = [f'{col}_{side}_{group}' for col, side, group in chart4.columns]
    chart4 = chart4.reset_index()

    # Ensure columns are numeric
    metric_side_type_cols = [f"{m}_{s}_{t}" for m in metric_cols for s in ["R", "L"] for t in ["FA", "OFF", "BB"]]
    missing_cols = [c for c in metric_side_type_cols if c not in chart4.columns]
    for col in missing_cols:
        chart4[col] = np.nan

    chart4[metric_side_type_cols] = chart4[metric_side_type_cols].apply(pd.to_numeric, errors="coerce")

    # Make sure correct team is assigned
    team_lookup = y.groupby('batter')['team'].first().rename('team')  
    chart4 = chart4.merge(team_lookup, on='batter', how='left')

    query2 = """
        SELECT
            * EXCLUDE (
                iz_raw_R_FA, iz_raw_L_FA, iz_raw_R_OFF, iz_raw_L_OFF, iz_raw_R_BB, iz_raw_L_BB,
                ooz_raw_R_FA, ooz_raw_L_FA, ooz_raw_R_OFF, ooz_raw_L_OFF, ooz_raw_R_BB, ooz_raw_L_BB,
                sd_raw_R_FA, sd_raw_L_FA, sd_raw_R_OFF, sd_raw_L_OFF, sd_raw_R_BB, sd_raw_L_BB
            ),
            ROUND(iz_raw_R_FA, 1) AS "z-swing% vs RHP FA",
            ROUND(iz_raw_L_FA, 1) AS "z-swing% vs LHP FA",
            ROUND(iz_raw_R_OFF, 1) AS "z-swing% vs RHP OFF",
            ROUND(iz_raw_L_OFF, 1) AS "z-swing% vs LHP OFF",
            ROUND(iz_raw_R_BB, 1) AS "z-swing% vs RHP BB",
            ROUND(iz_raw_L_BB, 1) AS "z-swing% vs LHP BB",
            ROUND(ooz_raw_R_FA, 1) AS "o-swing% vs RHP FA",
            ROUND(ooz_raw_L_FA, 1) AS "o-swing% vs LHP FA",
            ROUND(ooz_raw_R_OFF, 1) AS "o-swing% vs RHP OFF",
            ROUND(ooz_raw_L_OFF, 1) AS "o-swing% vs LHP OFF",
            ROUND(ooz_raw_R_BB, 1) AS "o-swing% vs RHP BB",
            ROUND(ooz_raw_L_BB, 1) AS "o-swing% vs LHP BB",
            ROUND(sd_raw_R_FA, 1) AS "sd% vs RHP FA",
            ROUND(sd_raw_L_FA, 1) AS "sd% vs LHP FA",
            ROUND(sd_raw_R_OFF, 1) AS "sd% vs RHP OFF",
            ROUND(sd_raw_L_OFF, 1) AS "sd% vs LHP OFF",
            ROUND(sd_raw_R_BB, 1) AS "sd% vs RHP BB",
            ROUND(sd_raw_L_BB, 1) AS "sd% vs LHP BB",
            CASE WHEN pitches_R_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_FA >= 50 ORDER BY iz_raw_R_FA) * 100) ELSE NULL END AS "z-swing vs RHP FA",
            CASE WHEN pitches_R_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_FA >= 50 ORDER BY ooz_raw_R_FA DESC) * 100) ELSE NULL END AS "o-swing vs RHP FA",
            CASE WHEN pitches_R_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_FA >= 50 ORDER BY sd_raw_R_FA) * 100) ELSE NULL END AS "sd vs RHP FA",
            CASE WHEN pitches_L_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_FA >= 50 ORDER BY iz_raw_L_FA) * 100) ELSE NULL END AS "z-swing vs LHP FA",
            CASE WHEN pitches_L_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_FA >= 50 ORDER BY ooz_raw_L_FA DESC) * 100) ELSE NULL END AS "o-swing vs LHP FA",
            CASE WHEN pitches_L_FA >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_FA >= 50 ORDER BY sd_raw_L_FA) * 100) ELSE NULL END AS "sd vs LHP FA",
            CASE WHEN pitches_R_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_OFF >= 50 ORDER BY iz_raw_R_OFF) * 100) ELSE NULL END AS "z-swing vs RHP OFF",
            CASE WHEN pitches_R_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_OFF >= 50 ORDER BY ooz_raw_R_OFF DESC) * 100) ELSE NULL END AS "o-swing vs RHP OFF",
            CASE WHEN pitches_R_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_OFF >= 50 ORDER BY sd_raw_R_OFF) * 100) ELSE NULL END AS "sd vs RHP OFF",
            CASE WHEN pitches_L_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_OFF >= 50 ORDER BY iz_raw_L_OFF) * 100) ELSE NULL END AS "z-swing vs LHP OFF",
            CASE WHEN pitches_L_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_OFF >= 50 ORDER BY ooz_raw_L_OFF DESC) * 100) ELSE NULL END AS "o-swing vs LHP OFF",
            CASE WHEN pitches_L_OFF >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_OFF >= 50 ORDER BY sd_raw_L_OFF) * 100) ELSE NULL END AS "sd vs LHP OFF",
            CASE WHEN pitches_R_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_BB >= 50 ORDER BY iz_raw_R_BB) * 100) ELSE NULL END AS "z-swing vs RHP BB",
            CASE WHEN pitches_R_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_BB >= 50 ORDER BY ooz_raw_R_BB DESC) * 100) ELSE NULL END AS "o-swing vs RHP BB",
            CASE WHEN pitches_R_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_R_BB >= 50 ORDER BY sd_raw_R_BB) * 100) ELSE NULL END AS "sd vs RHP BB",
            CASE WHEN pitches_L_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_BB >= 50 ORDER BY iz_raw_L_BB) * 100) ELSE NULL END AS "z-swing vs LHP BB",
            CASE WHEN pitches_L_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_BB >= 50 ORDER BY ooz_raw_L_BB DESC) * 100) ELSE NULL END AS "o-swing vs LHP BB",
            CASE WHEN pitches_L_BB >= 50 THEN ROUND(PERCENT_RANK() OVER (PARTITION BY pitches_L_BB >= 50 ORDER BY sd_raw_L_BB) * 100) ELSE NULL END AS "sd vs LHP BB"
        FROM (
            SELECT
                *,
                strike_swing_R_FA * 100.0 / NULLIF(strike_swing_R_FA + strike_taken_R_FA, 0) AS iz_raw_R_FA,
                strike_swing_L_FA * 100.0 / NULLIF(strike_swing_L_FA + strike_taken_L_FA, 0) AS iz_raw_L_FA,
                strike_swing_R_OFF * 100.0 / NULLIF(strike_swing_R_OFF + strike_taken_R_OFF, 0) AS iz_raw_R_OFF,
                strike_swing_L_OFF * 100.0 / NULLIF(strike_swing_L_OFF + strike_taken_L_OFF, 0) AS iz_raw_L_OFF,
                strike_swing_R_BB * 100.0 / NULLIF(strike_swing_R_BB + strike_taken_R_BB, 0) AS iz_raw_R_BB,
                strike_swing_L_BB * 100.0 / NULLIF(strike_swing_L_BB + strike_taken_L_BB, 0) AS iz_raw_L_BB,
                ball_swing_R_FA * 100.0 / NULLIF(ball_swing_R_FA + ball_taken_R_FA, 0) AS ooz_raw_R_FA,
                ball_swing_L_FA * 100.0 / NULLIF(ball_swing_L_FA + ball_taken_L_FA, 0) AS ooz_raw_L_FA,
                ball_swing_R_OFF * 100.0 / NULLIF(ball_swing_R_OFF + ball_taken_R_OFF, 0) AS ooz_raw_R_OFF,
                ball_swing_L_OFF * 100.0 / NULLIF(ball_swing_L_OFF + ball_taken_L_OFF, 0) AS ooz_raw_L_OFF,
                ball_swing_R_BB * 100.0 / NULLIF(ball_swing_R_BB + ball_taken_R_BB, 0) AS ooz_raw_R_BB,
                ball_swing_L_BB * 100.0 / NULLIF(ball_swing_L_BB + ball_taken_L_BB, 0) AS ooz_raw_L_BB,
                correct_sd_R_FA * 100.0 / NULLIF(pitches_R_FA, 0) AS sd_raw_R_FA,
                correct_sd_L_FA * 100.0 / NULLIF(pitches_L_FA, 0) AS sd_raw_L_FA,
                correct_sd_R_OFF * 100.0 / NULLIF(pitches_R_OFF, 0) AS sd_raw_R_OFF,
                correct_sd_L_OFF * 100.0 / NULLIF(pitches_L_OFF, 0) AS sd_raw_L_OFF,
                correct_sd_R_BB * 100.0 / NULLIF(pitches_R_BB, 0) AS sd_raw_R_BB,
                correct_sd_L_BB * 100.0 / NULLIF(pitches_L_BB, 0) AS sd_raw_L_BB
            FROM chart4
        )
    """

    chart4 = duckdb.sql(query2).df()

    chart4["pitches"] = round(
        chart4["pitches_R_FA"]
        .add(chart4["pitches_L_FA"], fill_value=0)
        .add(chart4["pitches_R_OFF"], fill_value=0)
        .add(chart4["pitches_L_OFF"], fill_value=0)
        .add(chart4["pitches_R_BB"], fill_value=0)
        .add(chart4["pitches_L_BB"], fill_value=0)
    )


    chart4 = chart4[['batter', 'player', 'bats', 'team', 'PA', 'pitches', 
                     'z-swing% vs RHP FA', 'o-swing% vs RHP FA', 'sd% vs RHP FA', 
                     'z-swing% vs RHP OFF', 'o-swing% vs RHP OFF', 'sd% vs RHP OFF', 
                     'z-swing% vs RHP BB', 'o-swing% vs RHP BB', 'sd% vs RHP BB',
                     'z-swing vs RHP FA', 'o-swing vs RHP FA', 'sd vs RHP FA', 
                     'z-swing vs RHP OFF', 'o-swing vs RHP OFF', 'sd vs RHP OFF',
                     'z-swing vs RHP BB', 'o-swing vs RHP BB', 'sd vs RHP BB',
                     'z-swing% vs LHP FA', 'o-swing% vs LHP FA', 'sd% vs LHP FA', 
                     'z-swing% vs LHP OFF', 'o-swing% vs LHP OFF', 'sd% vs LHP OFF', 
                     'z-swing% vs LHP BB', 'o-swing% vs LHP BB', 'sd% vs LHP BB',
                     'z-swing vs LHP FA', 'o-swing vs LHP FA', 'sd vs LHP FA', 
                     'z-swing vs LHP OFF', 'o-swing vs LHP OFF', 'sd vs LHP OFF',
                     'z-swing vs LHP BB', 'o-swing vs LHP BB', 'sd vs LHP BB']]
    
    chart4 = chart4.sort_values("PA", ascending=False)
    return chart4

# This function builds a where clause for our sql queries
def build_where_clause(count, outs, bases, phand, pitch_type):
    conditions = []

    if count:
        formatted = ", ".join(f"'{c}'" for c in count)
        conditions.append(f'"count" IN ({formatted})')

    if outs:
        formatted = ", ".join(str(o) for o in outs)
        conditions.append(f"outs_when_up IN ({formatted})")

    if bases:
        if "first" in bases:
            conditions.append("on_1b IS NOT NULL")
        if "second" in bases:
            conditions.append("on_2b IS NOT NULL")
        if "third" in bases:
            conditions.append("on_3b IS NOT NULL")
        if "not first" in bases:
            conditions.append("on_1b IS NULL")
        if "not second" in bases:
            conditions.append("on_2b IS NULL")
        if "not third" in bases:
            conditions.append("on_3b IS NULL")
        if "risp" in bases:
            conditions.append("(on_2b IS NOT NULL OR on_3b IS NOT NULL)")
        if "bases empty" in bases:
            conditions.append("(on_1b IS NULL AND on_2b IS NULL AND on_3b IS NULL)")

    if phand:
        conditions.append(f"p_throws = '{phand}'")

    if pitch_type:
        formatted = ", ".join(f"'{p}'" for p in pitch_type)
        conditions.append(f"pitch_type IN ({formatted})")

    if not conditions:
        return "1=1"
    return " AND ".join(conditions)

label_style = {"fontSize": "12px", "fontWeight": "normal", "color": "#4d4d4d", "marginBottom": "4px", "marginLeft": "4px", "display": "block"}

col_text = {
    "z-swing%": "In-Zone Swing %",
    "o-swing%": "Out-Of-Zone Swing %",
    "sd%": "Overall Swing Decision %",

    "z-swing% vs RHP": "In-Zone Swing % vs RHP",
    "z-swing% vs LHP": "In-Zone Swing % vs LHP",
    "o-swing% vs RHP": "Out-Of-Zone Swing % vs RHP",
    "o-swing% vs LHP": "Out-Of-Zone Swing % vs LHP",
    "sd% vs RHP": "Swing Decision % vs RHP",
    "sd% vs LHP": "Swing Decision % vs LHP",

    "z-swing% vs FA": "In-Zone Swing % vs Fastballs",
    "z-swing% vs OFF": "In-Zone Swing % vs Offspeed",
    "z-swing% vs BB": "In-Zone Swing % vs Breaking Balls",
    "o-swing% vs FA": "Out-Of-Zone Swing % vs Fastballs",
    "o-swing% vs OFF": "Out-Of-Zone Swing % vs Offspeed",
    "o-swing% vs BB": "Out-Of-Zone Swing % vs Breaking Balls",
    "sd% vs FA": "Swing Decision % vs Fastballs",
    "sd% vs OFF": "Swing Decision % vs Offspeed",
    "sd% vs BB": "Swing Decision % vs Breaking Balls",

    "z-swing% vs RHP FA": "In-Zone Swing % vs RHP Fastballs",
    "z-swing% vs LHP FA": "In-Zone Swing % vs LHP Fastballs",
    "z-swing% vs RHP OFF": "In-Zone Swing % vs RHP Offspeed",
    "z-swing% vs LHP OFF": "In-Zone Swing % vs LHP Offspeed",
    "z-swing% vs RHP BB": "In-Zone Swing % vs RHP Breaking Balls",
    "z-swing% vs LHP BB": "In-Zone Swing % vs LHP Breaking Balls",
    "o-swing% vs RHP FA": "Out-Of-Zone Swing % vs RHP Fastballs",
    "o-swing% vs LHP FA": "Out-Of-Zone Swing % vs LHP Fastballs",
    "o-swing% vs RHP OFF": "Out-Of-Zone Swing % vs RHP Offspeed",
    "o-swing% vs LHP OFF": "Out-Of-Zone Swing % vs LHP Offspeed",
    "o-swing% vs RHP BB": "Out-Of-Zone Swing % vs RHP Breaking Balls",
    "o-swing% vs LHP BB": "Out-Of-Zone Swing % vs LHP Breaking Balls",
    "sd% vs RHP FA": "Swing Decision % vs RHP Fastballs",
    "sd% vs LHP FA": "Swing Decision % vs LHP Fastballs",
    "sd% vs RHP OFF": "Swing Decision % vs RHP Offspeed",
    "sd% vs LHP OFF": "Swing Decision % vs LHP Offspeed",
    "sd% vs RHP BB": "Swing Decision % vs RHP Breaking Balls",
    "sd% vs LHP BB": "Swing Decision % vs LHP Breaking Balls",

    "z-swing": "In-Zone Swing Percentile",
    "o-swing": "Out-Of-Zone Swing Percentile",
    "sd": "Overall Swing Decision Percentile",
    
    "z-swing vs RHP": "In-Zone Swing vs RHP Percentile",
    "z-swing vs LHP": "In-Zone Swing vs LHP Percentile",
    "o-swing vs RHP": "Out-Of-Zone Swing vs RHP Percentile",
    "o-swing vs LHP": "Out-Of-Zone Swing vs LHP Percentile",
    "sd vs RHP": "Swing Decision vs RHP Percentile",
    "sd vs LHP": "Swing Decision vs LHP Percentile",
    
    "z-swing vs FA": "In-Zone Swing vs Fastballs Percentile",
    "z-swing vs OFF": "In-Zone Swing vs Offspeed Percentile",
    "z-swing vs BB": "In-Zone Swing vs Breaking Balls Percentile",
    "o-swing vs FA": "Out-Of-Zone Swing vs Fastballs Percentile",
    "o-swing vs OFF": "Out-Of-Zone Swing vs Offspeed Percentile",
    "o-swing vs BB": "Out-Of-Zone Swing vs Breaking Balls Percentile",
    "sd vs FA": "Swing Decision vs Fastballs Percentile",
    "sd vs OFF": "Swing Decision vs Offspeed Percentile",
    "sd vs BB": "Swing Decision vs Breaking Balls Percentile",
    
    "z-swing vs RHP FA": "In-Zone Swing vs RHP Fastballs Percentile",
    "z-swing vs LHP FA": "In-Zone Swing vs LHP Fastballs Percentile",
    "z-swing vs RHP OFF": "In-Zone Swing vs RHP Offspeed Percentile",
    "z-swing vs LHP OFF": "In-Zone Swing vs LHP Offspeed Percentile",
    "z-swing vs RHP BB": "In-Zone Swing vs RHP Breaking Balls Percentile",
    "z-swing vs LHP BB": "In-Zone Swing vs LHP Breaking Balls Percentile",
    "o-swing vs RHP FA": "Out-Of-Zone Swing vs RHP Fastballs Percentile",
    "o-swing vs LHP FA": "Out-Of-Zone Swing vs LHP Fastballs Percentile",
    "o-swing vs RHP OFF": "Out-Of-Zone Swing vs RHP Offspeed Percentile",
    "o-swing vs LHP OFF": "Out-Of-Zone Swing vs LHP Offspeed Percentile",
    "o-swing vs RHP BB": "Out-Of-Zone Swing vs RHP Breaking Balls Percentile",
    "o-swing vs LHP BB": "Out-Of-Zone Swing vs LHP Breaking Balls Percentile",
    "sd vs RHP FA": "Swing Decision vs RHP Fastballs Percentile",
    "sd vs LHP FA": "Swing Decision vs LHP Fastballs Percentile",
    "sd vs RHP OFF": "Swing Decision vs RHP Offspeed Percentile",
    "sd vs LHP OFF": "Swing Decision vs LHP Offspeed Percentile",
    "sd vs RHP BB": "Swing Decision vs RHP Breaking Balls Percentile",
    "sd vs LHP BB": "Swing Decision vs LHP Breaking Balls Percentile"
}

# Create app layout
app.layout = html.Div([

    # Header
    html.H1("MLB Hitters Swing Decision Dashboard", style={"fontFamily": "Helvetica", "marginBottom": "10px"}),

    # Filters header
    html.H4("Filters", style={"fontFamily": "Helvetica", "marginBottom": "5px"}),
    
    # Displaying all options for table filter
    html.Div([

        # Dropdown option to select swing decision split
        html.Div([
            html.Label("Split by", style=label_style),
            dcc.Dropdown(
                id="split",
                options=[
                    {"label": "Default", "value": "default"},
                    {"label": "Pitcher Handedness", "value": "pitcher_hand"},
                    {"label": "Pitch Type", "value": "pitch_group"},
                    {"label": "Pitcher Handedness + Pitch Type", "value": "hand_group"},
                ],
                value="default",
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to select style mode
        html.Div([
            html.Label("Style", style=label_style),
            dcc.Dropdown(
                id="style",
                options=[
                    {"label": "Simple", "value": "simple"},
                    {"label": "Detailed", "value": "detailed"},
                ],
                value="simple",
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to select view mode
        html.Div([
            html.Label("View", style=label_style),
            dcc.Dropdown(
                id="view",
                options=[
                    {"label": "Standard", "value": "standard"},
                    {"label": "Percentile", "value": "percentile"},
                ],
                value="standard",
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to select a team
        html.Div([
            html.Label("Team", style=label_style),
            dcc.Dropdown(
                id="team",
                options=[{"label": t, "value": t} for t in sorted(df["batting_team"].unique())],
                placeholder="Choose",
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to select batter hand
        html.Div([
            html.Label("Batter Hand", style=label_style),
            dcc.Dropdown(
                id="hand",
                options=[
                    {"label": "Right", "value": "R"},
                    {"label": "Left", "value": "L"},
                    {"label": "Switch", "value": "S"},
                ],
                placeholder="Choose",
                style={"width": "200px"}
            ),
        ]),

        html.Div([
            # Dropdown option to select PA minimum
            html.Label("PA Minimum", style=label_style),
            dcc.Dropdown(
                id="pa",
                options = [{"label": str(n), "value": n} for n in range(0, 550, 50)],
                placeholder="Choose",
                style={"width": "200px"}
            ),
        ]),
        
    ], style={"display": "flex", "gap": "20px", "marginBottom": "10px"}),

    # Customization label
    html.H4("Customization", style={"fontFamily": "Helvetica", "marginBottom": "5px"}),

    # Display customization choices
    html.Div([

        # Dropdown option to customize pitch count
        html.Div([
            html.Label("Count", style=label_style),
            dcc.Dropdown(
                id="count",
                options=["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"],
                placeholder="Choose",
                multi=True,
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to customize outs
        html.Div([
            html.Label("Outs", style=label_style),
            dcc.Dropdown(
                id="outs",
                options=[
                    {"label": "0", "value": 0},
                    {"label": "1", "value": 1},
                    {"label": "2", "value": 2}
                ],
                placeholder="Choose",
                multi=True,
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to customize base runners
        html.Div([
            html.Label("Base Runners", style=label_style),
            dcc.Dropdown(
                id="bases",
                options=[
                    {"label": "RISP", "value": "risp"},
                    {"label": "Bases Empty", "value": "bases empty"},
                    {"label": "First", "value": "first"},
                    {"label": "Second", "value": "second"},
                    {"label": "Third", "value": "third"},
                    {"label": "Not First", "value": "not first"},
                    {"label": "Not Second", "value": "not second"},
                    {"label": "Not Third", "value": "not third"},
                ],
                placeholder="Choose",
                multi=True,
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to customize pitcher hand
        html.Div([
            html.Label("Pitcher Hand", style=label_style),
            dcc.Dropdown(
                id="phand",
                options=[
                    {"label": "Right", "value": "R"},
                    {"label": "Left", "value": "L"},
                ],
                placeholder="Choose",
                style={"width": "200px"}
            ),
        ]),

        # Dropdown option to customize pitch type
        html.Div([
            html.Label("Pitch Type(s)", style=label_style),
            dcc.Dropdown(
                id="pitch_type",
                options=[
                    {"label": "Four-Seam Fastball", "value": "FF"},
                    {"label": "Sinker", "value": "SI"},
                    {"label": "Cutter", "value": "FC"},
                    {"label": "Slider", "value": "SL"},
                    {"label": "Sweeper", "value": "ST"},
                    {"label": "Curveball", "value": "CU"},
                    {"label": "Knuckle Curve", "value": "KC"},
                    {"label": "Slurve", "value": "SV"},
                    {"label": "Slow Curve", "value": "CS"},
                    {"label": "Changeup", "value": "CH"},
                    {"label": "Splitter", "value": "FS"},
                    {"label": "Forkball", "value": "FO"},
                    {"label": "Knuckleball", "value": "KN"},
                    {"label": "Eephus", "value": "EP"},
                ],
                placeholder="Choose",
                multi=True,
                style={"width": "200px"}
            ),
        ]),
        
    ], style={"display": "flex", "gap": "20px", "marginBottom": "10px"}),

    html.Div([
        dash_table.DataTable(
            id="table",
            columns=[{"name": c.upper(), "id": c} for c in df.columns],
            tooltip_header=col_text,
            css=[{"selector": ".dash-table-tooltip", "rule": "width: fit-content !important; max-width: 400px !important; min-width: 0px !important; white-space: nowrap !important;"}],
            tooltip_delay=0,
            tooltip_duration=None,
            fixed_columns={'headers': True, 'data': 1},
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            sort_action="native",
            style_header={
                "backgroundColor": "#1a1a1a",
                "color": "white",
                "fontWeight": "bold",
                "textAlign": "center"
            },
            style_cell={
                "textAlign": "center",
                "padding": "8px",
                "fontFamily": "Helvetica"
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f7f7f7"}
            ]
        )
    ])    
], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 20px"})

@app.callback(
    Output("table", "data"),
    Output("table", "columns"),
    Input("count", "value"),
    Input("outs", "value"),
    Input("bases", "value"),
    Input("phand", "value"),
    Input("pitch_type", "value"),
    Input("split", "value"),
    Input("style", "value"),
    Input("view", "value"),
    Input("team", "value"),
    Input("hand", "value"),
    Input("pa", "value")
)
def update_table(count, outs, bases, phand, pitch_type, split, style, view, team, hand, pa):

    # Default view
    if split == "default":
        result = default(count, outs, bases, phand, pitch_type)

        if style == "simple":
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd%"]]
        else:
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "z-swing", "o-swing", "sd"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "z-swing%", "o-swing%", "sd%"]]

    # Split by pitcher hand
    elif split == "pitcher_hand":
        result = pitcher_hand(count, outs, bases, phand, pitch_type)

        if style == "simple":
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd vs RHP", "sd vs LHP"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd% vs RHP", "sd% vs LHP"]]
        else:
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "z-swing vs RHP", "o-swing vs RHP", "sd vs RHP", "z-swing vs LHP", "o-swing vs LHP", "sd vs LHP"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "z-swing% vs RHP", "o-swing% vs RHP", "sd% vs RHP", "z-swing% vs LHP", "o-swing% vs LHP", "sd% vs LHP"]]

    # Split by pitch group
    elif split == "pitch_group":
        result = pitch_group(count, outs, bases, phand, pitch_type)

        if style == "simple":
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd vs FA", "sd vs OFF", "sd vs BB"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd% vs FA", "sd% vs OFF", "sd% vs BB"]]
        else:
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches",
                                 "z-swing vs FA", "o-swing vs FA", "sd vs FA",
                                 "z-swing vs OFF", "o-swing vs OFF", "sd vs OFF",
                                 "z-swing vs BB", "o-swing vs BB", "sd vs BB"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches",
                                 "z-swing% vs FA", "o-swing% vs FA", "sd% vs FA",
                                 "z-swing% vs OFF", "o-swing% vs OFF", "sd% vs OFF",
                                 "z-swing% vs BB", "o-swing% vs BB", "sd% vs BB"]]

    # Split by pitcher hand + pitch group
    elif split == "hand_group":
        result = hand_group(count, outs, bases, phand, pitch_type)

        if style == "simple":
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd vs RHP FA", "sd vs LHP FA", "sd vs RHP OFF", "sd vs LHP OFF", "sd vs RHP BB", "sd vs LHP BB"]]
            else:
                result = result[["batter", "player", "bats", "team", "PA", "pitches", "sd% vs RHP FA", "sd% vs LHP FA", "sd% vs RHP OFF", "sd% vs LHP OFF", "sd% vs RHP BB", "sd% vs LHP BB"]]
        else:
            if view == "percentile":
                result = result[[
                    "batter", "player", "bats", "team", "PA", "pitches",
                    "z-swing vs RHP FA", "o-swing vs RHP FA", "sd vs RHP FA",
                    "z-swing vs LHP FA", "o-swing vs LHP FA", "sd vs LHP FA",
                    "z-swing vs RHP OFF", "o-swing vs RHP OFF", "sd vs RHP OFF",
                    "z-swing vs LHP OFF", "o-swing vs LHP OFF", "sd vs LHP OFF",
                    "z-swing vs RHP BB", "o-swing vs RHP BB", "sd vs RHP BB",
                    "z-swing vs LHP BB", "o-swing vs LHP BB", "sd vs LHP BB"
                ]]
            else:
                result = result[[
                    "batter", "player", "bats", "team", "PA", "pitches",
                    "z-swing% vs RHP FA", "o-swing% vs RHP FA", "sd% vs RHP FA",
                    "z-swing% vs LHP FA", "o-swing% vs LHP FA", "sd% vs LHP FA",
                    "z-swing% vs RHP OFF", "o-swing% vs RHP OFF", "sd% vs RHP OFF",
                    "z-swing% vs LHP OFF", "o-swing% vs LHP OFF", "sd% vs LHP OFF",
                    "z-swing% vs RHP BB", "o-swing% vs RHP BB", "sd% vs RHP BB",
                    "z-swing% vs LHP BB", "o-swing% vs LHP BB", "sd% vs LHP BB"
                ]]

    # Team filter
    if team:
        result = result[result["team"] == team]

    # Batter handedness filter
    if hand:
        result = result[result["bats"] == hand]

    # PA minimum filter
    if pa:
        result = result[result["PA"] >= pa]

    # Drop ID from final result
    result = result.drop(columns=['batter'])

    columns = [{"name": c.upper(), "id": c} for c in result.columns]
    return result.to_dict("records"), columns

if __name__ == "__main__":
    app.run(debug=True)







