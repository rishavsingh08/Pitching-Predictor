# PitchPredictor


Overview

This is an algorithm that predicts the best possible pitch for a pitcher to throw to a batter in professional baseball. The algorithm is trained on Google Statcast Data from 2015-2024. It uses two LightGBM classifiers, one to predict swing decisions, and the other to predict optimal pitch type. The latter is trained to minimize expected batting average (x_BA). Variables for this model are biomechanical (height, weight, stance, etc.), historical (player’s career performance against similar pitchers), and atmospheric (weather, elevation, etc). Also, the model accounts for tunneling, the technique in which pitchers can use earlier pitches from an at-bat to greatly alter a batter’s swing decisions.

Additionally, the model improves after every use. Data from each at-bat that the model predicts is loaded into the existing LightGBM model. Matchup data is available to see on the CSV file ‘matchup.csv’ The model will refer to this data for repeat pitcher-batter matchups and similar scenarios.   

Usage

Run the function hs_ab(). The algorithm will ask the user to enter atmospheric conditions for the game.
The algorithm will then begin to ask for player specific details. Eventually, the algorithm will begin asking about the pitcher's arsenal. It will first for all their pitch types, and then average velocity for each pitch type. Even if the algorithm has seen this pitcher before, it will ask these questions, in case of pitch type/ pitch velocity variations.
The first pitch is up to the user’s discretion. The algorithm will display this message at the conclusion of the introductory questions, and the user will be asked to enter details about that pitch.
Assuming the at-bat continues after the first pitch, the model will begin producing recommendations. It provides a zone and a pitch type for every recommendation. Continue following the display’s prompts throughout the at-bat
At the conclusion of the at-bat, data will be loaded into the existing LightGBM model. Data from the at-bat will also be reflected in ‘matchup.csv’. The model is now ready for a new at-bat.


Important Notes

Data from this model is derived from Baseball Savant which contains all MLB Statcast data from the last 10 years:  https://baseballsavant.mlb.com/
After downloading, the data was thoroughly processed, cleaned and organized. However, the training data is not available on GitHub as it exceeded the size limit (it’s trained on over 5 million pitches). 
All models needed for this project are saved as pkl files, so the project works perfectly fine without having the data on hand
Pitch Zone Number Reference Guide: https://medium.com/@thomasjamesnestico/classifying-mlb-pitch-zones-and-predicting-milb-zones-7e95cf308254
Pitch Type Number Reference Guide: https://www.mlb.com/glossary/pitch-types Refer to the pitch map in the code for how these pitches should be referred to during usage.
For more on x_BA and why my model is trained to minimize it: https://www.mlb.com/glossary/statcast/expected-batting-average
Please read ‘requirements.txt’ to ensure the user is ready to use the model
Thank You for checking this out! Feel free to reach out for any questions about the project or baseball in general. 




