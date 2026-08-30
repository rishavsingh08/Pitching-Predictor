import pandas as pd
from pybaseball import statcast
from pybaseball import playerid_reverse_lookup
from pybaseball import cache
from datetime import datetime, timedelta
from tqdm import tqdm
cache.enable()

game_df = statcast(start_dt="2025-03-27", end_dt="2025-07-01")
# game_df=game_df[game_df['game_pk']== 777940]


positive_outcomes = [
    'single',
    'double',
    'triple',
    'home_run',
    'walk',
    'intent_walk',
    'hit_by_pitch',
    'field_error',
    'catcher_interf',
    'fan_interference',
    'batter_interference',  # Only if batter reaches base (edge case)
    'sac_fly',
    'sac_bunt'
]



keep_cols=['pitch_type', 'release_speed', 'player_name',
           'batter', 'events', 'description', 
           'zone', 'stand', 'p_throws', 'balls',
           'strikes', 'pitch_number', 'estimated_ba_using_speedangle', 'game_date']

game_df=game_df[keep_cols]
game_df['game_date'] = pd.to_datetime(game_df['game_date'])

#Numerical values for pitches and side


pitch_map = {
    'FF': 1,   # Four Seam Fastball
    'SI': 2,   # Sinker
    'FC': 3,   # Cutter
    'IN': 4,

    'FS': 5,   # Splitter
    'SL': 6,   # Slider
    'CU': 7,   # Curveball
    'KC': 8,   # Knuckle Curve
    'SC': 9,   # Screwball
    'SV': 10,   # Slurve
    'ST': 11,  # Sweeper
    'FO': 12,  # Forkball
    'KN': 13,  # Knuckleball

    'CH': 14,  # Changeup
    'EP': 15   # Eepush
}

side_map = {
    'L': 0,  # Left
    'R': 1   # Right
}

game_df['pitch_type'] = game_df['pitch_type'].map(pitch_map)
game_df['p_throws'] = game_df['p_throws'].map(side_map)
game_df['stand'] = game_df['stand'].map(side_map)


game_df['prev_pitch_type'] = game_df['pitch_type'].shift(-1)
game_df['prev_velocity'] = game_df['release_speed'].shift(-1)

# Drop the last row, which now contains NaNs from the shift
game_df = game_df.iloc[:-1]


#Historical Data (Matchup and Career)

def download_in_chunks(start_date, end_date, freq='90D'):
    all_data = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current <= end:
        next_date = min(current + pd.to_timedelta(freq), end)
        try:
            print(f"Downloading {current.date()} to {next_date.date()}...")
            chunk = statcast(start_dt=str(current.date()), end_dt=str(next_date.date()))
            all_data.append(chunk)
        except Exception as e:
            print(f"Failed: {current.date()} to {next_date.date()} | {e}")
        current = next_date + timedelta(days=1)

    return pd.concat(all_data, ignore_index=True)

def pitch_group_averages(curr_df):
    bb_total = bb_p = 0
    os_total = os_p = 0
    fb_total = fb_p = 0
    for _, pitch_row in curr_df.iterrows():
        pitch_type = int(pitch_row['pitch_type'])
        event = pitch_row['events']

        # Fastballs (1–3)
        if pitch_type < 5.0:
            fb_total += 1
            if event in positive_outcomes:
                fb_p += 1

        # Breaking balls (4–12)
        if 5.0 <= pitch_type <= 13.0:
            bb_total += 1
            if event in positive_outcomes:
                bb_p += 1

        # Off-speed (13–14)
        if pitch_type >= 14.0:
            os_total += 1
            if event in positive_outcomes:
                os_p += 1
    
    return [
        fb_p / fb_total if fb_total > 0 else 0,
        bb_p / bb_total if bb_total > 0 else 0,
        os_p / os_total if os_total > 0 else 0,
        fb_total
    ]

def pa_counter(curr_df, curr_game_date):
    curr_df=curr_df[curr_df['game_date']<curr_game_date]
    return int(len(curr_df))
    
    

hist_df = pd.read_parquet("hist_df.parquet")
hist_df['game_date'] = pd.to_datetime(hist_df['game_date'])


hist_df['pitch_type'] = hist_df['pitch_type'].map(pitch_map)
hist_df = hist_df.dropna(subset=['pitch_type'])




m_os_hist=[]
m_bb_hist=[]
m_fb_hist=[]
m_fb_counter=[]

c_os_hist=[]
c_bb_hist=[]
c_fb_hist=[]
c_fb_counter=[]

matchup_pa=[]


for i, game_row in tqdm(list(game_df.iterrows()), desc="Processing Matchups"):
    matchup_df = hist_df[
        (hist_df['batter'] == game_row['batter']) &
        (hist_df['player_name'] == game_row['player_name'])
    ]
    career_df=hist_df[hist_df['batter'] == game_row['batter']]
    
    
    matchup_df = matchup_df.dropna(subset=['events'])
    matchup_df = matchup_df[~matchup_df['events'].isin(['walk', 'intent_walk', 'hit_by_pitch'])]
    
    
    career_df = career_df.dropna(subset=['events'])
    matchup_pa.append(pa_counter(matchup_df, game_row['game_date']))
    career_df = career_df[~career_df['events'].isin(['walk', 'intent_walk', 'hit_by_pitch'])]
    
    
    matchup_hist=pitch_group_averages(matchup_df)
    career_hist=pitch_group_averages(career_df)
    
    m_fb_hist.append(matchup_hist[0])
    m_bb_hist.append(matchup_hist[1])
    m_os_hist.append(matchup_hist[2])
    m_fb_counter.append(matchup_hist[3])
    
    c_fb_hist.append(career_hist[0])
    c_bb_hist.append(career_hist[1])
    c_os_hist.append(career_hist[2])
    c_fb_counter.append(career_hist[3])



game_df['MHOSBA']=m_os_hist
game_df['MHFBBA']=m_fb_hist
game_df['MHBBBA']=m_bb_hist
game_df['MFBC']= m_fb_counter

game_df['pa_number'] = matchup_pa

game_df['CHOSBA']=c_os_hist
game_df['CHFBBA']=c_fb_hist
game_df['CHBBBA']=c_bb_hist
game_df['CFBC']= c_fb_counter

   
# Batter names

bid_list=game_df['batter'].to_list()
b_names=[]
for bid in bid_list:
    result = playerid_reverse_lookup([bid])
    full_name= result.iloc[0]['name_first'] + " " + result.iloc[0]['name_last']
    b_names.append(full_name)

game_df['batter']=b_names 

#Removing balls/walks/hit by pitches

game_df = game_df[game_df['description'].isin(['called_strike', 'swinging_strike', 'hit_into_play','swinging_strike_blocked'])]
game_df=game_df[game_df['pitch_number'] !=1]

# filling expected batting averages

game_df['estimated_ba_using_speedangle'] = game_df['estimated_ba_using_speedangle'].fillna(0.0)

#Dropping unneeded columns

game_df=game_df.drop(['events', 'description', 'game_date', 'CFBC', 'MFBC'], axis=1)
game_df.to_parquet("u_games.parquet", index=False)

game_df.to_csv('u_games.csv', index=False)