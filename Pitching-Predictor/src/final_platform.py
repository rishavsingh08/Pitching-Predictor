import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from services.launch_angle import find_xBA

features= ["pitch_type", "release_speed", 'zone',"p_throws",
            "stand" , "balls","strikes","MHFBBA","MHBBBA","MHOSBA",
            "pitch_number",'prev_pitch_type','prev_velocity','pa_number']
            

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

matchup_df=pd.read_csv('matchup.csv')

def hs_ab():
    temp=int(input("What is the temperature of the game?"))
    elev=int(input("What is the elevation of the field?"))
    
    p_name=input("Batter Name: ")
    b_name=input("Pitcher Name: ")
    
    p_name=p_name.title()
    b_name=b_name.title()
    
    pitch_list= ask_pitch_list()
    pitch_vels= ask_pitch_vels(pitch_list)
    
    curr_df = matchup_df[(matchup_df['b_name'] == b_name) & (matchup_df['p_name'] == p_name)]
    
    if len(curr_df)==0:
        curr_df= new_matchup(p_name, b_name)
        
    hist= curr_df.iloc[0] 
    
    ab_results=pitch_by_pitch(temp, elev, hist, pitch_list, pitch_vels)
    
    idx = matchup_df.index[(matchup_df['p_name'] == p_name) & (matchup_df['b_name'] == b_name)]

    if isinstance(ab_results, pd.Series):
        matchup_df.loc[idx[0]] = ab_results.reindex(matchup_df.columns)
    else:
        update_model(ab_results[0])

        matchup_df.loc[idx[0]] = ab_results[1].reindex(matchup_df.columns)

    
    
    matchup_df.to_csv('matchup.csv', index=False)
    
    
def new_matchup(p_name, b_name):
    p_throws=int(input('Pitcher Dominant Arm: 1 for Right and 0 for Left: '))
    stand=int(input('Batter Side: 1 for Righty and 0 for Lefty: '))
    
    new_matchup_dict = {
    "p_name": p_name,
    "b_name": b_name,
    "p_throws": p_throws,
    "stand" : stand,
    "fb_abs": 0,
    "fb_hits": 0,
    "bb_abs": 0,
    "bb_hits": 0,
    "os_abs": 0,
    "os_hits": 0,
    }
    
    return_df= pd.DataFrame([new_matchup_dict])
    
    global matchup_df
    
    matchup_df=pd.concat([matchup_df, return_df], axis=0, ignore_index=True)
    
    return return_df

def ask_pitch_list():
    print("refer to pitch map for answering the following questions")
    
    pitch_list=[]
    first_pitch=int(input("What is the first pitch"))
    
    pitch_list.append(first_pitch)
    
    while True:
        new_pitch=int(input("Current List: "+list_displayer(pitch_list)+'\nWhat is the Next Pitch: '))
        pitch_list.append(new_pitch)
        ans= input("Current List: "+list_displayer(pitch_list)+'\nAny other pitches? ')
        
        if ans.strip().lower() == "no" :
            break
    
    return pitch_list
    
def ask_pitch_vels(pitch_list):
    pitch_types = [k for k, v in pitch_map.items() if v in pitch_list]
    pitch_vel_list=[]
    
    for i, pitch in enumerate(pitch_types):
        pitch_vel_list.append(float(input("Average Pitch Velocity of Pitcher's "+str(pitch)+"? : ")))
    
    return pitch_vel_list
        
def list_displayer(list):
    output= " ".join(str(x) for x in list)
    return output
      
def first_pitch(hist):
    print("Throw any pitch you'd like for the first pitch")
    
    first_pitch_type=float(input("First Pitch Type: "))
    
    first_pitch_vel=float(input('First Pitch Velocity: '))
    
    ans=input("Did the AB end after the first pitch")
    
    if ans.strip().lower()== "yes":
        ip_ans= input("Was the ball put in play")
        
        if ip_ans.strip().lower()=='yes':
            hit_ans=input("Was the batted ball a hit?")
            
            if first_pitch_type< 5:
                hist['fb_abs']= hist['fb_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['fb_hits']= hist['fb_hits']+1
            
            if first_pitch_type>4 and first_pitch_type<14:
                hist['bb_abs']= hist['bb_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['bb_hits']= hist['bb_hits']+1
                    
            if first_pitch_type>13:
                hist['os_abs']= hist['os_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['os_hits']= hist['os_hits']+1
                    
        return hist
    
    bs= input("Was the first pitch a ball or a strike?")
    
    
    return [first_pitch_type, first_pitch_vel, bs]
    
def df_setup(pitch_type, pitch_vel, zone, balls, strikes, pitch_number, prev_pitch_type, prev_velocity, hist, xBA):
    
    output_dict = {
    "pitch_type": pitch_type,
    "release_speed": pitch_vel,
    'zone': zone,
    "p_throws": hist['p_throws'],
    "stand" : hist['stand'],
    "balls": balls,
    "strikes": strikes,
    "MHFBBA": hist['fb_hits']/hist['fb_abs']  if hist['fb_abs'] > 0 else 0,
    "MHBBBA": hist['bb_hits']/hist['bb_abs'] if hist['bb_abs'] > 0 else 0,
    "MHOSBA": hist['os_hits']/hist['os_abs'] if hist['os_abs'] > 0 else 0,
    "pitch_number": pitch_number,
    'prev_pitch_type':prev_pitch_type,
    'prev_velocity':prev_velocity,
    'pa_number': hist['fb_abs']+hist['os_abs']+hist['bb_abs'],
    'estimated_ba_using_speedangle': xBA
    }
    
    output_df=pd.DataFrame([output_dict])
    
    return output_df

def pitch_by_pitch(temp, elev, hist, pitch_list, pitch_vels):
    ab_pitch_list=[]
    
    strikes = balls = 0
    
    fpl= first_pitch(hist)
    
    if isinstance(fpl, pd.Series):
        return fpl
    
    if str(fpl[2].strip().lower())== 'ball':
        balls+=1
    else:
        strikes+=1
       
    prev_pitch_type=fpl[0]
    prev_vel=fpl[1]
    pitch_number=1
    
    while True:
        next_pitch= predict_pitch(pitch_list, pitch_vels, hist, balls, strikes, prev_pitch_type, prev_vel, pitch_number)
        
        xBA=0.0
        
        print("Next Pitch Type: " + str(next_pitch[0])+ "\nNext Pitch Zone: "+str(next_pitch[1]))
        
        acc_list=acc_data()
        ip_ans= input("Was the ball put in play")
        hit_ans=input("Was the batted ball a hit?")
        
        if ip_ans.strip().lower()=='yes':
            est_dist=int(input("How far was the ball hit (first time it hit the ground)?"))
            mph_ans = int(input(" Type 0 for a weak hit, 1 for an average hit, 2 for a hard hit, and 3 for an extremely hard hit"))
            type= int(input("Type 1 for Line Drive or Ground Ball or Type 2 for Fly Ball or Pop Fly")) 
            
            if mph_ans == 0:
                ev=60
            if mph_ans == 1:
                ev=75
            if mph_ans == 2:
                ev=85
            if mph_ans == 3:
                ev=95
                
            xBA= find_xBA(ev, est_dist, temp, elev, type)
            
            if next_pitch[0]< 5:
                hist['fb_abs']= hist['fb_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['fb_hits']= hist['fb_hits']+1
            
            if next_pitch[0]>4 and next_pitch[0]<14:
                hist['bb_abs']= hist['bb_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['bb_hits']= hist['bb_hits']+1
                    
            if next_pitch[0]>13:
                hist['os_abs']= hist['os_abs']+1
                if hit_ans.strip().lower()== 'yes':
                    hist['os_hits']= hist['os_hits']+1
                
            
            
            ab_pitch_list.append(df_setup(next_pitch[0],acc_list[0], acc_list[1], balls, strikes, pitch_number, prev_pitch_type, prev_vel, hist, xBA ))
            
            
            break
        
        bs_ans = int(input("Type 1 for Ball, 2 for Strike, 3 for Foul"))
        
        if bs_ans==1:
            balls+=1
            if balls==4:
                break
        else:
            ab_pitch_list.append(df_setup(next_pitch[0],acc_list[0], acc_list[1], balls, strikes, pitch_number, prev_pitch_type, prev_vel, hist, 0.0 ))
            if bs_ans==2:
                strikes+=1
                if strikes==3:
                    if next_pitch[0]< 5:
                        hist['fb_abs']= hist['fb_abs']+1
                    if next_pitch[0]>4 and next_pitch[0]<14:
                        hist['bb_abs']= hist['bb_abs']+1
                    if next_pitch[0]>=14:
                        hist['os_abs']= hist['os_abs']+1
                    
                    break
        
        term_ans=int(input("Type 1 if the AB has ended. 0 if it didn't"))
        if term_ans==1:
            break
        
        prev_pitch_type=next_pitch[0]
        prev_vel=acc_list[1]
        
    
    ab_pitch_df = pd.concat(ab_pitch_list, ignore_index=True)
    
    return [ab_pitch_df, hist]
            
            
def acc_data():
    
    mph = float(input("How hard was the pitch"))   
    
    zone = float(input("Which zone was the pitch located"))  

    return [mph, zone]



def predict_pitch(pitch_list, pitch_vels, hist, balls, strikes, prev_pitch_type, prev_vel, pitch_number):
    pitch_poss_list=[]
    
    for i in range(len(pitch_list)):
        for z in range(1,15):
            
            pitch_dict = {
                "pitch_type": pitch_list[i],
                "release_speed": pitch_vels[i],
                'zone': float(z),
                "p_throws": hist['p_throws'],
                "stand" : hist['stand'],
                "balls": balls,
                "strikes": strikes,
                "MHFBBA": hist['fb_hits']/hist['fb_abs']  if hist['fb_abs'] > 0 else 0,
                "MHBBBA": hist['bb_hits']/hist['bb_abs'] if hist['bb_abs'] > 0 else 0,
                "MHOSBA": hist['os_hits']/hist['os_abs'] if hist['os_abs'] > 0 else 0,
                "pitch_number": pitch_number,
                'prev_pitch_type':prev_pitch_type,
                'prev_velocity': prev_vel,
                'pa_number': hist['fb_abs']+hist['os_abs']+hist['bb_abs'],
                'estimated_ba_using_speedangle': np.nan
            }
            
            pitch_poss_list.append(pd.DataFrame([pitch_dict]))
    
    pitch_poss_df = pd.concat(pitch_poss_list, ignore_index=True)
    
    model = joblib.load('x_avg_lightgbm.pkl')
    
    nb_model= joblib.load('not_ball_prob.pkl')
    
    
    xBA_predictions= model.predict(pitch_poss_df[features])
    
    pitch_poss_df['estimated_ba_using_speedangle']= xBA_predictions
    
    nb_predictions= nb_model.predict(pitch_poss_df[['pitch_type', 'release_speed', 
                                                    'zone', 'stand', 'p_throws', 'balls',
                                                    'strikes','pitch_number',
                                                    'prev_pitch_type', 'prev_velocity']])
    
    pitch_poss_df['bs']= nb_predictions
    
    
    pitch_poss_df=pitch_poss_df[pitch_poss_df['bs']==1]
    
    min_row = pitch_poss_df.loc[pitch_poss_df['estimated_ba_using_speedangle'].idxmin()]

    best_pitch_type = min_row['pitch_type']
    best_zone = min_row['zone']
    
    return [best_pitch_type,best_zone]


def update_model(training_df):

    model = joblib.load('x_avg_lightgbm_tester.pkl')


    X_new = training_df[features]
    y_new = training_df['estimated_ba_using_speedangle']
    train_data_new = lgb.Dataset(X_new, label=y_new)


    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data_new,
        num_boost_round=50,       
        init_model=model
    )

    joblib.dump(model, 'x_avg_lightgbm_tester.pkl')



    





