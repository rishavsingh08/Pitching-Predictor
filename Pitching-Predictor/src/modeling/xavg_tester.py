#ruff: noqa
import pandas as pd
import joblib
from pycaret.regression import *  

df=pd.read_csv("/Users/rishavsingh/Pitching Predictor/Pitching-Predictor/data/processed/u_games.csv")

df=df.drop(['player_name', 'batter', 'CHOSBA','CHFBBA', 'CHBBBA'], axis=1)

e = setup(df, target = 'estimated_ba_using_speedangle', session_id = 123)  

best=compare_models()

print(best)

evaluate_model(best)

plot_model(best, plot = 'residuals')

plot_model(best, plot = 'feature')


predict_model(best)

predictions = predict_model(best, data=df)

joblib.dump(best, 'x_avg_lightgbm_tester.pkl')

