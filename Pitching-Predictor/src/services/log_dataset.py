import pandas as pd
from pybaseball import statcast
from pybaseball import cache

cache.enable()

df = pd.read_parquet("hist_df.parquet")


keep_cols=['pitch_type', 'release_speed', 'description', 
           'zone', 'stand', 'p_throws', 'balls',
           'strikes', 'pitch_number']

log_df=df[keep_cols]


def des_sorter(value):
    if value in ['ball', 'hit_by_pitch']:
        return 0
    else:
        return 1
    
log_df['description']= log_df['description'].apply(des_sorter)

pitch_map = {
    'FF': 1,   # Four Seam Fastball
    'SI': 2,   # Sinker
    'FC': 3,   # Cutter

    'FS': 4,   # Splitter
    'SL': 5,   # Slider
    'CU': 6,   # Curveball
    'KC': 7,   # Knuckle Curve
    'SC': 8,   # Screwball
    'SV': 9,   # Slurve
    'ST': 10,  # Sweeper
    'FO': 11,  # Forkball
    'KN': 12,  # Knuckleball

    'CH': 13,  # Changeup
    'EP': 14   # Eepush
}

side_map = {
    'L': 0,  # Left
    'R': 1   # Right
}

log_df['pitch_type'] = log_df['pitch_type'].map(pitch_map)
log_df['p_throws'] = log_df['p_throws'].map(side_map)
log_df['stand'] = log_df['stand'].map(side_map)

log_df['prev_pitch_type'] = log_df['pitch_type'].shift(-1)
log_df['prev_velocity'] = log_df['release_speed'].shift(-1)

log_df=log_df[log_df['pitch_number'] !=1]




log_df.to_csv('log.csv', index= False)





