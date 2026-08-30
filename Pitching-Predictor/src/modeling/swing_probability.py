import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df= pd.read_csv('/Users/rishavsingh/Pitching Predictor/Pitching-Predictor/data/processed/log.csv')

X = df[['pitch_type', 'release_speed', 
           'zone', 'stand', 'p_throws', 'balls',
           'strikes','pitch_number',
           'prev_pitch_type', 'prev_velocity']]

y = df['description']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_pred, y_test)

print(accuracy)

joblib.dump(clf, 'not_ball_prob.pkl')

