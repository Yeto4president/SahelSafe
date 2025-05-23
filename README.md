# SahelSafe
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-blue?logo=pandas)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=xgboost)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikit-learn)
![Folium](https://img.shields.io/badge/Folium-0.14%2B-green?logo=map)
![Plotly](https://img.shields.io/badge/Plotly-5.10%2B-green?logo=plotly)
## Introduction
**SahelSafe** is a machine learning project aimed at predicting high-risk zones and periods for terrorist attacks in Burkina Faso, using open ACLED data. Leveraging an XGBoost model, interactive visualizations, and a "Data for Good" approach, this project seeks to provide alerts and risk maps to support security and humanitarian efforts in the Sahel region. Last updated: May 23, 2025, 11:20 PM CEST.


### A Mission Born from Personal Commitment
I was born in Burkina Faso. It's always been a peaceful country. Until the day when, in total incomprehension, terrorists started attacking certain parts of the country and even once the capital. For a while it didn't affect me directly, but some of my relatives were affected by the death of several of their acquaintances in the attacks.
This crisis has several consequences: thousands of people are fleeing danger, the economy is stalled by instability and an impatient population is attacking its leaders. Above all, thousands of families are shattered by the loss of their loved ones.

 
When I began exploring conflict data in Burkina Faso, I didn’t just see numbers and coordinates—I saw *lives impacted*, communities uprooted, and an urgent need for solutions. Terrorism, which has shaken my country in recent years, is more than a security challenge; it’s a human crisis that demands action. This predictive analysis project was born from that realization. I wanted to use my data science skills not for an academic exercise, but to contribute, however modestly, to *changing things*.

### A "Data for Good" Approach
Armed with ACLED’s open data and a passion for societal impact, I built a machine learning model to forecast high-risk zones and periods for attacks. It’s not just code or a map—it’s an attempt to give decision-makers, NGOs, or local communities tools to anticipate, prepare, and perhaps *save lives*. By integrating interactive visualizations and probability-based alerts, I aimed to turn raw data into actionable resources.

### My Ambition: Making a Difference
This project is a first step. My goal is clear: to leverage technology to address real challenges in my country and beyond. Whether refining this model with new data (e.g., climate or displacement metrics) or deploying it for practical use, I’m driven by the desire to create a *positive impact*. I invite anyone sharing this vision—developers, data scientists, or humanitarians—to contribute or inspire me to go further.
## Algorithms Used

### XGBoost (eXtreme Gradient Boosting)
XGBoost is the core algorithm used for prediction in this project. It’s a tree-based machine learning algorithm that employs gradient boosting to iteratively improve predictions:
- **How It Works**: XGBoost builds multiple decision trees sequentially. Each tree corrects the errors of the previous ones by minimizing a loss function (log-loss in this case). The final prediction is a weighted combination of all trees.
- **Why XGBoost?**: It excels with tabular data, handles imbalanced datasets well, and captures complex interactions between features like `cumulative_attacks` and `attack_count_prev_week`. It’s also fast and scalable.
- **Implementation**: The model is trained with parameters like `random_state=42` for reproducibility and `eval_metric='logloss'` to optimize for binary classification.

### SMOTE (Synthetic Minority Oversampling Technique)
SMOTE is used to address the class imbalance in the dataset, as weeks without attacks far outnumber those with attacks:
- **How It Works**: SMOTE generates synthetic samples for the minority class (weeks with attacks) by interpolating between existing minority samples. This balances the dataset, ensuring the model doesn’t over-prioritize the majority class.
- **Why SMOTE?**: It helps improve the model’s ability to predict rare events (attacks), which is critical for this project’s goal of not missing potential risks.
- **Implementation**: Applied only to the training set to avoid data leakage, using the `imblearn` library.

### Visualization Libraries
- **Folium**: Used to create interactive maps (e.g., `burkina_risk_map.html`) showing attack locations and predicted risk probabilities by region. It leverages Leaflet.js for rendering.
- **Plotly**: Used for time-series visualizations (e.g., `burkina_predicted_proba.html`) to plot trends like attack frequency or predicted probabilities over time.
## Installation
- Install the required dependencies:
  ```bash
  pip install pandas numpy xgboost scikit-learn imbalanced-learn folium plotly geopy joblib
  ```
- Place the data files (e.g., acled_burkina_aggregated.csv) in the root directory.
## Files
- clean_acled_data.py: Script to clean the ACLED data and generate initial visualizations.
- model_training.py: Script to train and evaluate the XGBoost model.
- acled_burkina_cleaned.csv: Cleaned dataset with all original columns and added features.
- acled_burkina_aggregated.csv: Aggregated data by region and week, used for modeling.
- burkina_risk_map.html: Interactive map of predicted risk probabilities.
- burkina_predicted_proba.html: Graph of predicted probabilities over time.
- xgb_model.pkl: Saved trained XGBoost model for reuse.
- burkina_attacks_map.html: Map of historical attack locations.
- burkina_attacks_timeseries.html: Time series graph of attack trends.

## How to Launch the Analysis
Prepare the Environment:
- Ensure all dependencies are installed (see Installation section).
- Download the raw ACLED data for Burkina Faso (2015-2025) from https://acleddata.com/data-export-tool/ and save it as acled_burkina_2015_2025.csv in the root directory.
- Run Data Cleaning:
- Execute the cleaning script to process the raw data:
``` bash
python clean_acled_data.py
```
This generates acled_burkina_cleaned.csv, acled_burkina_aggregated.csv, burkina_attacks_map.html, and burkina_attacks_timeseries.html.
## Run Model Training:
Execute the modeling script to train the model and generate predictions:
``` bash
python model_training.py
```
This produces xgb_model.pkl, burkina_risk_map.html, and burkina_predicted_proba.html, along with evaluation metrics in the console.
### Explore Results:
- Open the .html files in a web browser to view the visualizations.
- Check the console output from model_training.py for performance metrics (e.g., AUC-ROC, precision).


##  Results
View the risk map: burkina_risk_map.html
![image](https://github.com/user-attachments/assets/83a69441-568c-40d0-a093-163a51b2b3c8)
Check evaluation metrics in the console after running model_training.py.
![image](https://github.com/user-attachments/assets/6226b7ae-5786-4251-9323-83cbe2c4b4c1)

Explanation of Predicted Probabilities Graph (2024-2025)

The "Probabilities predicted by region (2024-2025)" graph visualizes the XGBoost model's predicted likelihood of terrorist attacks across Burkina Faso's regions over weekly intervals. The x-axis shows weeks (1-50), the y-axis shows probabilities (0-1), and each colored line represents a region (e.g., Sahel, Est). Peaks (e.g., ~0.9 for Sahel around week 15) indicate high-risk periods, reflecting historical trends and features like is_rainy_season. This aids in identifying priority areas for intervention, though accuracy depends on data quality and unforeseen events.
## Contributing
Contributions are welcome! Feel free to fork this repository, submit issues, or propose enhancements (e.g., adding climate data or improving the model). Let’s work together to make this project more impactful.


Last Updated
May 23, 2025, 11:20 PM CEST
