import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import time


g = 9.81
Cd = 0.30
A = 0.00314
m = 0.145
R = 287.05
T0 = 288.15
P0 = 101325
L = 0.0065


positive_outcomes = [
    'single',
    'double',
    'triple',
    'home_run'
]

keep_cols=['description', 'events', 'launch_speed', 'launch_angle']

def feet_to_meters(ft): return ft * 0.3048
def meters_to_feet(m): return m / 0.3048
def mph_to_mps(mph): return mph * 0.44704
def fahrenheit_to_kelvin(F): return (F - 32) * 5 / 9 + 273.15


def get_air_density(temp_f, elevation_ft):
    T = fahrenheit_to_kelvin(temp_f)
    h = feet_to_meters(elevation_ft)
    P = P0 * (1 - L * h / T0) ** (g * 0.0289644 / (R * L))
    return P / (R * T)


def simulate_trajectory(angle_rad, v0, rho):
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)

    def derivatives(t, y):
        x, y_pos, vx, vy = y
        v = np.sqrt(vx**2 + vy**2)
        Fd = 0.5 * Cd * A * rho * v**2
        ax = -Fd * vx / (v * m)
        ay = -g - (Fd * vy / (v * m))
        return [vx, vy, ax, ay]

    y0 = [0, 1.0, vx0, vy0]

    def hit_ground(t, y): return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    # Larger max_step for speedup
    sol = solve_ivp(derivatives, [0, 10], y0, events=hit_ground, max_step=0.05)
    return sol.y[0, -1]


def get_distance_for_angle(angle_deg, exit_velocity_mph, temp_f, elevation_ft):
    angle_rad = np.radians(angle_deg)
    v0 = mph_to_mps(exit_velocity_mph)
    rho = get_air_density(temp_f, elevation_ft)
    dist_m = simulate_trajectory(angle_rad, v0, rho)
    return meters_to_feet(dist_m)



def try_find_launch_angle(ev_mph, target_ft, temp_f, elev_ft, angle_bracket):
    v0 = mph_to_mps(ev_mph)
    rho = get_air_density(temp_f, elev_ft)
    target_m = feet_to_meters(target_ft)

    def error(angle_deg):
        angle_rad = np.radians(angle_deg)
        sim_m = simulate_trajectory(angle_rad, v0, rho)
        return sim_m - target_m

    try:
        result = root_scalar(error, bracket=angle_bracket, method='brentq')
        if result.converged:
            return result.root
    except ValueError:
        pass
    return None


def find_launch_angle(ev_mph, target_ft, temp_f, elev_ft, delta=0.1, max_shift=5):
    """
    Finds both low and high launch angles (in degrees) that reach a given distance.

    Always returns a list with two angles:
    - [low, high] if both exist
    - [single, single] if only one solution exists
    - [45.0, 45.0] as fallback (neutral parabola)

    All inputs: ev_mph (mph), target_ft (ft), temp_f (°F), elev_ft (ft)
    """
    tested = set()
    low_angle = None
    high_angle = None

    for i in range(int(max_shift / delta) + 1):
        for direction in [+1, -1]:
            test_ev = round(ev_mph + direction * i * delta, 3)
            if test_ev <= 0 or test_ev in tested:
                continue
            tested.add(test_ev)

            if low_angle is None:
                low_angle = try_find_launch_angle(test_ev, target_ft, temp_f, elev_ft, angle_bracket=[-90, 30])

            if high_angle is None:
                high_angle = try_find_launch_angle(test_ev, target_ft, temp_f, elev_ft, angle_bracket=[20,90])

            if low_angle is not None and high_angle is not None:
                print("low/high: " + str(low_angle) + " / " + str(high_angle))
                return [low_angle, high_angle]

    if low_angle is not None:
        print("low/low: " + str(low_angle) + " / " + str(low_angle))
        return [low_angle, low_angle]
    
    elif high_angle is not None:
        print("high/high: " + str(high_angle) + " / " + str(high_angle))
        return [high_angle, high_angle]
    else:
        print("no no")
        return [20, 40]  # Fallback to neutral parabola



ip_df = pd.read_parquet('/Users/rishavsingh/Pitching Predictor/Pitching-Predictor/src/ip_df.parquet')



def find_xBA(ev_mph, target_ft, temp_f, elev_ft, type):
    
    la=la_chooser(find_launch_angle(ev_mph, target_ft, temp_f, elev_ft), type)
    
    amp= param_finder(int(ev_mph), la)
    
    curr_df = ip_df[(ip_df['launch_angle'].between(la - amp, la + amp)) & (ip_df['launch_speed'].between(ev_mph - amp, ev_mph + amp))]
    
    positive_df=curr_df[curr_df['events'].isin(positive_outcomes)]
    
    if len(curr_df) == 0:
        return 0.0
    
    return((len(positive_df)/len(curr_df)))
    
        
    
def param_finder(ev_mph, la):
    
    curr_df = ip_df[(ip_df['launch_angle'].round() == la) & (ip_df['launch_speed'].round() == ev_mph)]

    amp = 0

    if len(curr_df) >= 10:
        print("amp: " + str(amp))
        return amp

    while (len(curr_df) < 10)  and (amp<16):
        
        
        amp += 1

        curr_df = ip_df[
            (ip_df['launch_angle'].between(la - amp, la + amp)) &
            (ip_df['launch_speed'].between(ev_mph - amp, ev_mph + amp))
        ]

    print("amp: " + str(amp))
    return amp

def la_chooser(la_list, type):
    
    
    
    if type==1:
        if la_list[0]<=30:
            return round(la_list[0])
        else:
            return round(la_list[1])
        
    if type==2:
        if la_list[0]>20:
            return round(la_list[0])
        else:
            return round(la_list[1])



    
    
    
    
    