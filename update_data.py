from pybaseball import statcast, playerid_reverse_lookup, batting_stats_range
from datetime import date
import numpy as np

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

def clean_data(data, end_dt):
    ### ATTAIN PLAYER NAMES USING BATTER ID
    batter_ids = data['batter'].unique().tolist()
    lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    lookup['batter_name'] = lookup['name_first'] + ' ' + lookup['name_last']
    lookup['batter_name'] = lookup['batter_name'].str.title()
    data1 = data.merge(lookup[['key_mlbam', 'batter_name']], left_on='batter', right_on='key_mlbam', how='left')

    ### ASSIGN HITTING TEAMS
    data1['batting_team'] = data1.apply(
        lambda row: row['away_team'] if row['inning_topbot'] == 'Top' else row['home_team'],
        axis=1
    )

    ### SORT DATASET TO ONLY INCLUDE PITCHES WITH VALID OUTCOMES
    desc_valid = ['hit_into_play', 'called_strike', 'ball', 'swinging_strike',
           'blocked_ball', 'foul', 'foul_tip', 'hit_by_pitch',
           'swinging_strike_blocked', 'pitchout', 'swinging_pitchout',
            'missed_bunt', 'foul_bunt', 'bunt_foul_tip']
    data2 = data1[data1["description"].isin(desc_valid)]

    data2["count"] = data2["balls"].astype("str") + "-" + data2["strikes"].astype("str")

    fa = ['FF', 'SI', 'FC']
    bb = ['SL', 'ST', 'CU', 'KC', 'SV', 'CS']
    off = ['CH', 'FS', 'FO', 'SC']
    other = ['KN', 'EP', 'FA', 'UN']

    data2["pitch_group"] = np.nan
    data2.loc[data2["pitch_type"].isin(fa), "pitch_group"] = "FA"
    data2.loc[data2["pitch_type"].isin(bb), "pitch_group"] = "BB"
    data2.loc[data2["pitch_type"].isin(off), "pitch_group"] = "OFF"
    data2.loc[data2["pitch_type"].isin(other), "pitch_group"] = "OTHER"

    data3 = data2[['balls', 'strikes', 'count', 'outs_when_up', 'batting_team', 'batter_name', 'batter', 'stand', 'description', 'zone', 'pitch_type', 'pitch_group', 'p_throws', 'on_1b', 'on_2b', 'on_3b']]

    no_na = ['balls', 'strikes', 'count', 'outs_when_up', 'batting_team', 'batter_name', 'batter', 'stand', 'description', 'zone', 'pitch_type', 'pitch_group', 'p_throws']
    data3 = data3.dropna(subset=no_na)

    strikes = list(range(1,10))
    swing = ["hit_into_play", "swinging_strike", "foul", "foul_tip", "swinging_strike_blocked", "swinging_pitchout", 'missed_bunt', 'foul_bunt', 'bunt_foul_tip']

    data4 = data3.copy()

    data4["strike_swing"] = 0
    data4["strike_taken"] = 0
    data4["ball_swing"] = 0
    data4["ball_taken"] = 0

    data4.loc[(data4["zone"].isin(strikes)) & (data4["description"].isin(swing)), "strike_swing"] = 1
    data4.loc[(data4["zone"].isin(strikes)) & (~data4["description"].isin(swing)), "strike_taken"] = 1
    data4.loc[(~data4["zone"].isin(strikes)) & (data4["description"].isin(swing)), "ball_swing"] = 1
    data4.loc[(~data4["zone"].isin(strikes)) & (~data4["description"].isin(swing)), "ball_taken"] = 1

    data4["correct_sd"] = 0
    data4["incorrect_sd"] = 0
    data4.loc[(data4["strike_swing"] == 1) | (data4["ball_taken"] == 1), "correct_sd"] = 1
    data4.loc[(data4["strike_taken"] == 1) | (data4["ball_swing"] == 1), "incorrect_sd"] = 1

    for batter in list(data4["batter"].unique()):
        x = data4[data4["batter"] == batter]
        if len(x["stand"].unique()) == 2:
            data4.loc[data4["batter"] == batter, "stand"] = "S"
        else:
            continue

    szn_data = batting_stats_range(start_dt='2026-03-25', end_dt=end_dt)
    szn_data = szn_data[["PA", "mlbID"]]
    data5 = data4.merge(szn_data[['mlbID','PA']], left_on='batter', right_on='mlbID', how='left')

    return data5

end_dt = date.today().strftime('%Y-%m-%d')
data = statcast(start_dt='2026-03-25', end_dt=end_dt)
df = clean_data(data, end_dt)
df.to_parquet('season_data.parquet')