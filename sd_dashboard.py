# Dash imports
import dash
from dash import dcc, html, Input, Output, dash_table

# Pandas & numpy
import pandas as pd
import numpy as np

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
def default(df):

    # Group by just the batter
    chart1 = df.groupby(["batter"]).agg(
        player = ("batter_name", "first"),
        bats = ("stand", "first"),
        team = ("batting_team", "first"),
        PA = ("PA", "first"),
        correct_sd = ("correct_sd", "sum"),
        strike_swing = ("strike_swing", "sum"),
        strike_taken = ("strike_taken", "sum"),
        ball_swing = ("ball_swing", "sum"),
        ball_taken = ("ball_taken", "sum"),
        total = ("batter", "count")
    )
    chart1 = chart1.reset_index()

    # Define SD-related percentages
    chart1["iz%"] = round(chart1["strike_swing"] / (chart1["strike_swing"] + chart1["strike_taken"]) * 100, 1)
    chart1["ooz%"] = round(chart1["ball_taken"] / (chart1["ball_taken"] + chart1["ball_swing"]) * 100, 1)
    chart1["sd%"] = round(chart1["correct_sd"] / chart1["total"] * 100, 1)

    # Get percentiles for each SD metric
    eligible = chart1["total"] >= 150
    chart1["iz"] = round(chart1.loc[eligible, "iz%"].rank(pct=True) * 100)
    chart1["ooz"] = round(chart1.loc[eligible, "ooz%"].rank(pct=True) * 100)
    chart1["sd"] = round(chart1.loc[eligible, "sd%"].rank(pct=True) * 100)

    # Include only columns of interest
    chart1 = chart1[["batter", "player", "bats", "team", "PA", "iz%", "iz", "ooz%", "ooz", "sd%", "sd"]]

    # Sort by Correct Swing Decision %
    chart1 = chart1.sort_values("PA", ascending=False)

    # Return updated df
    return chart1

# This function returns the version of the dataframe that is split by the handedness of the opposing pitcher
def pitcher_hand(df):

    # Group by batter and pitcher handedness
    y = df.groupby(["batter", "p_throws"]).agg(
        player = ("batter_name", "first"),
        bats = ("stand", "first"),
        team = ("batting_team", "first"),
        PA = ("PA", "first"),
        correct_sd = ("correct_sd", "sum"),
        strike_swing = ("strike_swing", "sum"),
        strike_taken = ("strike_taken", "sum"),
        ball_swing = ("ball_swing", "sum"),
        ball_taken = ("ball_taken", "sum"),
        total = ("batter", "count")
    )
    y = y.reset_index()

    # Columns to attain for both right and left handed pitchers
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'total']

    # Pivot data
    chart2 = y.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns='p_throws',
        values=metric_cols
    )

    # Create a column for RHP and LHP with columns of interest
    chart2.columns = [f'{col}_{side}' for col, side in chart2.columns]
    chart2 = chart2.reset_index()

    # Ensure columns are numeric
    metric_side_cols = [f"{m}_{s}" for m in metric_cols for s in ["R", "L"]]
    chart2[metric_side_cols] = chart2[metric_side_cols].apply(pd.to_numeric, errors="coerce")

    # Make sure correct team is assigned
    team_lookup = df.groupby('batter')['batting_team'].first().rename('team')  
    chart2 = chart2.merge(team_lookup, on='batter', how='left')

    # Define SD-related percentages vs RHP and LHP
    for side in ["R", "L"]:
        chart2[f"iz% vs {side}HP"] = round(chart2[f"strike_swing_{side}"] / (chart2[f"strike_swing_{side}"] + chart2[f"strike_taken_{side}"]) * 100, 1)
        chart2[f"ooz% vs {side}HP"] = round(chart2[f"ball_taken_{side}"] / (chart2[f"ball_swing_{side}"] + chart2[f"ball_taken_{side}"]) * 100, 1)
        chart2[f"sd% vs {side}HP"] = round(chart2[f"correct_sd_{side}"] / chart2[f"total_{side}"] * 100, 1)

    # Get percentiles for each SD metric
    for side in ["R", "L"]:
        eligible = chart2[f"total_{side}"] >= 100
        chart2[f"iz vs {side}HP"] = round(chart2.loc[eligible, f"iz% vs {side}HP"].rank(pct=True) * 100)
        chart2[f"ooz vs {side}HP"] = round(chart2.loc[eligible, f"ooz% vs {side}HP"].rank(pct=True) * 100)
        chart2[f"sd vs {side}HP"] = round(chart2.loc[eligible, f"sd% vs {side}HP"].rank(pct=True) * 100)

    # Only include columns of interest
    chart2 = chart2[['batter', 'player', 'bats', 'team', 'PA', 'iz% vs RHP',
                     'ooz% vs RHP', 'sd% vs RHP', 'iz% vs LHP', 'ooz% vs LHP',
                     'sd% vs LHP', 'iz vs RHP', 'ooz vs RHP',
                     'sd vs RHP', 'iz vs LHP', 'ooz vs LHP', 'sd vs LHP']]

    # Sort by PA
    chart2 = chart2.sort_values("PA", ascending=False)

    # Return updated df
    return chart2

# This function returns the version of the dataframe that is split by the pitch type the batter is facing
def pitch_group(df):

    # Group by batter and pitch group
    w = df.groupby(["batter", "pitch_group"]).agg(
        player = ("batter_name", "first"),
        bats = ("stand", "first"),
        team = ("batting_team", "first"),
        PA = ("PA", "first"),
        correct_sd = ("correct_sd", "sum"),
        strike_swing = ("strike_swing", "sum"),
        strike_taken = ("strike_taken", "sum"),
        ball_swing = ("ball_swing", "sum"),
        ball_taken = ("ball_taken", "sum"),
        total = ("batter", "count")
    )
    w = w.reset_index()

    # Columns to attain for all pitch groups
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'total']

    # Pivot data
    chart3 = w.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns='pitch_group',
        values=metric_cols
    )

    # Create a column for FA, OFF, and BB with columns of interest
    chart3.columns = [f'{col}_{side}' for col, side in chart3.columns]
    chart3 = chart3.reset_index()

    # Ensure columns are numeric
    metric_type_cols = [f"{m}_{t}" for m in metric_cols for t in ["FA", "OFF", "BB"]]
    chart3[metric_type_cols] = chart3[metric_type_cols].apply(pd.to_numeric, errors="coerce")

    # Make sure correct team is assigned
    team_lookup = df.groupby('batter')['batting_team'].first().rename('team')  
    chart3 = chart3.merge(team_lookup, on='batter', how='left')

    # Define SD-related percentages vs FA, OFF, and BB
    for group in ["FA", "OFF", "BB"]:
        chart3[f"iz% vs {group}"] = round(chart3[f"strike_swing_{group}"] / (chart3[f"strike_swing_{group}"] + chart3[f"strike_taken_{group}"]) * 100, 1)
        chart3[f"ooz% vs {group}"] = round(chart3[f"ball_taken_{group}"] / (chart3[f"ball_swing_{group}"] + chart3[f"ball_taken_{group}"]) * 100, 1)
        chart3[f"sd% vs {group}"] = round(chart3[f"correct_sd_{group}"] / chart3[f"total_{group}"] * 100, 1)

    # Get percentiles for each SD metric
    for group in ["FA", "OFF", "BB"]:
        eligible = chart3[f"total_{group}"] >= 50
        chart3[f"iz vs {group}"] = round(chart3.loc[eligible, f"iz% vs {group}"].rank(pct=True) * 100)
        chart3[f"ooz vs {group}"] = round(chart3.loc[eligible, f"ooz% vs {group}"].rank(pct=True) * 100)
        chart3[f"sd vs {group}"] = round(chart3.loc[eligible, f"sd% vs {group}"].rank(pct=True) * 100)
    
    # Only include columns of interest
    chart3 = chart3[['batter', 'player', 'bats', 'team', 'PA', 'iz% vs FA',
                     'ooz% vs FA', 'sd% vs FA', 'iz% vs OFF', 'ooz% vs OFF',
                     'sd% vs OFF', 'iz% vs BB', 'ooz% vs BB', 'sd% vs BB',
                     'iz vs FA', 'ooz vs FA', 'sd vs FA',
                     'iz vs OFF', 'ooz vs OFF', 'sd vs OFF',
                     'iz vs BB', 'ooz vs BB', 'sd vs BB']]

    # Sort by PA
    chart3 = chart3.sort_values("PA", ascending=False)
    
    # Return updated df
    return chart3

def hand_group(df):

    # Group by batter, pitcher handedness, and pitch type
    z = df.groupby(["batter", "p_throws", "pitch_group"]).agg(
        player = ("batter_name", "first"),
        bats = ("stand", "first"),
        team = ("batting_team", "first"),
        PA = ("PA", "first"),
        correct_sd = ("correct_sd", "sum"),
        strike_swing = ("strike_swing", "sum"),
        strike_taken = ("strike_taken", "sum"),
        ball_swing = ("ball_swing", "sum"),
        ball_taken = ("ball_taken", "sum"),
        total = ("batter", "count")
    )
    z = z.reset_index()

    # Columns to attain for all pitcher handedness + pitch group combos
    metric_cols = ['correct_sd', 'strike_swing', 'strike_taken', 'ball_swing', 'ball_taken', 'total']

    # Pivot data
    chart4 = z.pivot(
        index=['batter', 'player', 'bats', 'PA'],
        columns=['p_throws', 'pitch_group'],
        values=metric_cols
    )

    # Create a column for each combo of RHP/LHP and pitch group with columns of interest
    chart4.columns = [f'{col}_{side}_{group}' for col, side, group in chart4.columns]
    chart4 = chart4.reset_index()

    # Ensure columns are numeric
    metric_side_type_cols = [f"{m}_{s}_{t}" for m in metric_cols for s in ["R", "L"] for t in ["FA", "OFF", "BB"]]
    chart4[metric_side_type_cols] = chart4[metric_side_type_cols].apply(pd.to_numeric, errors="coerce")

    # Make sure correct team is assigned
    team_lookup = df.groupby('batter')['batting_team'].first().rename('team')  
    chart4 = chart4.merge(team_lookup, on='batter', how='left')

    # Define SD-related percentages for each possible combo
    for side in ["R", "L"]:
        for group in ["FA", "OFF", "BB"]:
            suffix1 = f"vs {side}HP {group}"
            suffix2 = f"{side}_{group}"
            chart4[f"iz% {suffix1}"] = round(chart4[f"strike_swing_{suffix2}"] / (chart4[f"strike_swing_{suffix2}"] + chart4[f"strike_taken_{suffix2}"]) * 100, 1)
            chart4[f"ooz% {suffix1}"] = round(chart4[f"ball_taken_{suffix2}"] / (chart4[f"ball_swing_{suffix2}"] + chart4[f"ball_taken_{suffix2}"]) * 100, 1)
            chart4[f"sd% {suffix1}"] = round(chart4[f"correct_sd_{suffix2}"] / chart4[f"total_{suffix2}"] * 100, 1)

    # Get percentiles for each SD metric
    for side in ["R", "L"]:
        for group in ["FA", "OFF", "BB"]:
            eligible = chart4[f"total_{side}_{group}"] >= 50
            chart4[f"iz vs {side}HP {group}"] = round(chart4.loc[eligible, f"iz% vs {side}HP {group}"].rank(pct=True) * 100)
            chart4[f"ooz vs {side}HP {group}"] = round(chart4.loc[eligible, f"ooz% vs {side}HP {group}"].rank(pct=True) * 100)
            chart4[f"sd vs {side}HP {group}"] = round(chart4.loc[eligible, f"sd% vs {side}HP {group}"].rank(pct=True) * 100)

    # Only include columns of interest
    chart4 = chart4[['batter', 'player', 'bats', 'team', 'PA', 
                     'iz% vs RHP FA', 'ooz% vs RHP FA', 'sd% vs RHP FA', 
                     'iz% vs LHP FA', 'ooz% vs LHP FA', 'sd% vs LHP FA', 
                     'iz% vs RHP OFF', 'ooz% vs RHP OFF', 'sd% vs RHP OFF', 
                     'iz% vs LHP OFF', 'ooz% vs LHP OFF', 'sd% vs LHP OFF', 
                     'iz% vs RHP BB', 'ooz% vs RHP BB', 'sd% vs RHP BB', 
                     'iz% vs LHP BB', 'ooz% vs LHP BB', 'sd% vs LHP BB',
                     'iz vs RHP FA', 'ooz vs RHP FA', 'sd vs RHP FA', 
                     'iz vs LHP FA', 'ooz vs LHP FA', 'sd vs LHP FA', 
                     'iz vs RHP OFF', 'ooz vs RHP OFF', 'sd vs RHP OFF', 
                     'iz vs LHP OFF', 'ooz vs LHP OFF', 'sd vs LHP OFF', 
                     'iz vs RHP BB', 'ooz vs RHP BB', 'sd vs RHP BB', 
                     'iz vs LHP BB', 'ooz vs LHP BB', 'sd vs LHP BB'
                    ]]

    # Sort by PA
    chart4 = chart4.sort_values("PA", ascending=False)

    # Return updated df
    return chart4

label_style = {"fontSize": "12px", "fontWeight": "normal", "color": "#4d4d4d", "marginBottom": "4px", "marginLeft": "4px", "display": "block"}

col_text = {
    "iz%": "In-Zone Swing %",
    "ooz%": "Out-Of-Zone Take %",
    "sd%": "Overall Swing Decision %",

    "iz% vs RHP": "In-Zone Swing % vs RHP",
    "iz% vs LHP": "In-Zone Swing % vs LHP",
    "ooz% vs RHP": "Out-Of-Zone Take % vs RHP",
    "ooz% vs LHP": "Out-Of-Zone Take % vs LHP",
    "sd% vs RHP": "Swing Decision % vs RHP",
    "sd% vs LHP": "Swing Decision % vs LHP",

    "iz% vs FA": "In-Zone Swing % vs Fastballs",
    "iz% vs OFF": "In-Zone Swing % vs Offspeed",
    "iz% vs BB": "In-Zone Swing % vs Breaking Balls",
    "ooz% vs FA": "Out-Of-Zone Take % vs Fastballs",
    "ooz% vs OFF": "Out-Of-Zone Take % vs Offspeed",
    "ooz% vs BB": "Out-Of-Zone Take % vs Breaking Balls",
    "sd% vs FA": "Swing Decision % vs Fastballs",
    "sd% vs OFF": "Swing Decision % vs Offspeed",
    "sd% vs BB": "Swing Decision % vs Breaking Balls",

    "iz% vs RHP FA": "In-Zone Swing % vs RHP Fastballs",
    "iz% vs LHP FA": "In-Zone Swing % vs LHP Fastballs",
    "iz% vs RHP OFF": "In-Zone Swing % vs RHP Offspeed",
    "iz% vs LHP OFF": "In-Zone Swing % vs LHP Offspeed",
    "iz% vs RHP BB": "In-Zone Swing % vs RHP Breaking Balls",
    "iz% vs LHP BB": "In-Zone Swing % vs LHP Breaking Balls",
    "ooz% vs RHP FA": "Out-Of-Zone Take % vs RHP Fastballs",
    "ooz% vs LHP FA": "Out-Of-Zone Take % vs LHP Fastballs",
    "ooz% vs RHP OFF": "Out-Of-Zone Take % vs RHP Offspeed",
    "ooz% vs LHP OFF": "Out-Of-Zone Take % vs LHP Offspeed",
    "ooz% vs RHP BB": "Out-Of-Zone Take % vs RHP Breaking Balls",
    "ooz% vs LHP BB": "Out-Of-Zone Take % vs LHP Breaking Balls",
    "sd% vs RHP FA": "Swing Decision % vs RHP Fastballs",
    "sd% vs LHP FA": "Swing Decision % vs LHP Fastballs",
    "sd% vs RHP OFF": "Swing Decision % vs RHP Offspeed",
    "sd% vs LHP OFF": "Swing Decision % vs LHP Offspeed",
    "sd% vs RHP BB": "Swing Decision % vs RHP Breaking Balls",
    "sd% vs LHP BB": "Swing Decision % vs LHP Breaking Balls",

    "iz": "In-Zone Swing Percentile",
    "ooz": "Out-Of-Zone Take Percentile",
    "sd": "Overall Swing Decision Percentile",
    
    "iz vs RHP": "In-Zone Swing vs RHP Percentile",
    "iz vs LHP": "In-Zone Swing vs LHP Percentile",
    "ooz vs RHP": "Out-Of-Zone Take vs RHP Percentile",
    "ooz vs LHP": "Out-Of-Zone Take vs LHP Percentile",
    "sd vs RHP": "Swing Decision vs RHP Percentile",
    "sd vs LHP": "Swing Decision vs LHP Percentile",
    
    "iz vs FA": "In-Zone Swing vs Fastballs Percentile",
    "iz vs OFF": "In-Zone Swing vs Offspeed Percentile",
    "iz vs BB": "In-Zone Swing vs Breaking Balls Percentile",
    "ooz vs FA": "Out-Of-Zone Take vs Fastballs Percentile",
    "ooz vs OFF": "Out-Of-Zone Take vs Offspeed Percentile",
    "ooz vs BB": "Out-Of-Zone Take vs Breaking Balls Percentile",
    "sd vs FA": "Swing Decision vs Fastballs Percentile",
    "sd vs OFF": "Swing Decision vs Offspeed Percentile",
    "sd vs BB": "Swing Decision vs Breaking Balls Percentile",
    
    "iz vs RHP FA": "In-Zone Swing vs RHP Fastballs Percentile",
    "iz vs LHP FA": "In-Zone Swing vs LHP Fastballs Percentile",
    "iz vs RHP OFF": "In-Zone Swing vs RHP Offspeed Percentile",
    "iz vs LHP OFF": "In-Zone Swing vs LHP Offspeed Percentile",
    "iz vs RHP BB": "In-Zone Swing vs RHP Breaking Balls Percentile",
    "iz vs LHP BB": "In-Zone Swing vs LHP Breaking Balls Percentile",
    "ooz vs RHP FA": "Out-Of-Zone Take vs RHP Fastballs Percentile",
    "ooz vs LHP FA": "Out-Of-Zone Take vs LHP Fastballs Percentile",
    "ooz vs RHP OFF": "Out-Of-Zone Take vs RHP Offspeed Percentile",
    "ooz vs LHP OFF": "Out-Of-Zone Take vs LHP Offspeed Percentile",
    "ooz vs RHP BB": "Out-Of-Zone Take vs RHP Breaking Balls Percentile",
    "ooz vs LHP BB": "Out-Of-Zone Take vs LHP Breaking Balls Percentile",
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
                options = [{"label": str(n), "value": n} for n in range(0, 450, 50)],
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
    x = df.copy()

    # Customize counts to be included
    if count:
        x = x[x["count"].isin(count)]

    # Customize outs to be included
    if outs:
        x = x[x["outs_when_up"].isin(outs)]

    # Customize base runner situations to be included
    if bases:
        mask = pd.Series(True, index=x.index)
        if "first" in bases:
            mask &= x["on_1b"].notna()
        if "second" in bases:
            mask &= x["on_2b"].notna()
        if "third" in bases:
            mask &= x["on_3b"].notna()
        if "not first" in bases:
            mask &= x["on_1b"].isna()
        if "not second" in bases:
            mask &= x["on_2b"].isna()
        if "not third" in bases:
            mask &= x["on_3b"].isna()
        if "risp" in bases:
            mask &= x["on_2b"].notna() | x["on_3b"].notna()
        if "bases empty" in bases:
            mask &= x["on_1b"].isna() & x["on_2b"].isna() & x["on_3b"].isna()
        x = x[mask]

    # Customize opposing pitcher handedness to be included
    if phand:
        x = x[x["p_throws"] == phand]

    # Customize pitch type(s) to be included
    if pitch_type:
        x = x[x["pitch_type"].isin(pitch_type)]

    # Default view
    if split == "default":
        result = default(x)

        # Simple view
        if style == "simple":
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "sd"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "sd%"]]
        # Detailed view    
        else:
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "iz", "ooz", "sd"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "iz%", "ooz%", "sd%"]]

    
    # Split by pitcher hand
    elif split == "pitcher_hand":
        result = pitcher_hand(x)

        # Simple view
        if style == "simple":
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "sd vs RHP", "sd vs LHP"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "sd% vs RHP", "sd% vs LHP"]]
        # Detailed view
        else:
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "iz vs RHP", "ooz vs RHP", "sd vs RHP", "iz vs LHP", "ooz vs LHP", "sd vs LHP"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "iz% vs RHP", "ooz% vs RHP", "sd% vs RHP", "iz% vs LHP", "ooz% vs LHP", "sd% vs LHP"]]

    # Split by pitch group
    elif split == "pitch_group":
        result = pitch_group(x)

        # Simple view
        if style == "simple":
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "sd vs FA", "sd vs OFF", "sd vs BB"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "sd% vs FA", "sd% vs OFF", "sd% vs BB"]]
        # Detailed view
        else:
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", 
                                 "iz vs FA", "ooz vs FA", "sd vs FA",
                                 "iz vs OFF", "ooz vs OFF", "sd vs OFF",
                                 "iz vs BB", "ooz vs BB", "sd vs BB"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA",
                                 "iz% vs FA", "ooz% vs FA", "sd% vs FA", 
                                 "iz% vs OFF", "ooz% vs OFF", "sd% vs OFF", 
                                 "iz% vs BB", "ooz% vs BB", "sd% vs BB"]]

    # Split by pitcher hand + pitch group
    elif split == "hand_group":
        result = hand_group(x)

        # Simple view
        if style == "simple":
            # Percentile view
            if view == "percentile":
                result = result[["batter", "player", "bats", "team", "PA", "sd vs RHP FA", "sd vs LHP FA", "sd vs RHP OFF", "sd vs LHP OFF", "sd vs RHP BB", "sd vs LHP BB"]]
            # Standard view
            else:
                result = result[["batter", "player", "bats", "team", "PA", "sd% vs RHP FA", "sd% vs LHP FA", "sd% vs RHP OFF", "sd% vs LHP OFF", "sd% vs RHP BB", "sd% vs LHP BB"]]
        # Detailed view
        else:
            # Percentile view
            if view == "percentile":
                result = result[[
                    "batter", "player", "bats", "team", "PA",
                    "iz vs RHP FA", "ooz vs RHP FA", "sd vs RHP FA", 
                    "iz vs LHP FA", "ooz vs LHP FA", "sd vs LHP FA", 
                    "iz vs RHP OFF", "ooz vs RHP OFF", "sd vs RHP OFF", 
                    "iz vs LHP OFF", "ooz vs LHP OFF", "sd vs LHP OFF", 
                    "iz vs RHP BB", "ooz vs RHP BB", "sd vs RHP BB", 
                    "iz vs LHP BB", "ooz vs LHP BB", "sd vs LHP BB"
                ]]
            # Standard view
            else:
                result = result[[
                    "batter", "player", "bats", "team", "PA",
                    "iz% vs RHP FA", "ooz% vs RHP FA", "sd% vs RHP FA", 
                    "iz% vs LHP FA", "ooz% vs LHP FA", "sd% vs LHP FA", 
                    "iz% vs RHP OFF", "ooz% vs RHP OFF", "sd% vs RHP OFF", 
                    "iz% vs LHP OFF", "ooz% vs LHP OFF", "sd% vs LHP OFF", 
                    "iz% vs RHP BB", "ooz% vs RHP BB", "sd% vs RHP BB", 
                    "iz% vs LHP BB", "ooz% vs LHP BB", "sd% vs LHP BB"
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

    # Add pitch count column only when a customization is applied
    if count or outs or bases or phand or pitch_type:
        pitch_counts = x.groupby("batter").size().rename("pitches")
        result = result.merge(pitch_counts, left_on="batter", right_index=True, how="left")

        cols = result.columns.tolist()
        cols.remove("pitches")
        cols.insert(5, "pitches")
        result = result[cols]

    # Drop ID from final result
    result = result.drop(columns=['batter'])

    columns = [{"name": c.upper(), "id": c} for c in result.columns]
    return result.to_dict("records"), columns

if __name__ == "__main__":
    app.run(debug=True)







