# Nadia.Project
Nadia Maarfavi - 100730530
The project uses machine learning models to predict movie opening weekend sales based on trailer content, especially the actors' facial features. This repository contains the code for the final report and the machine-learning models used for the project.
Contents
The repository consists of the following files and folders:
•	Report.pdf: This file contains a detailed report on the project, including the motivation behind the project, the dataset used, the methodology employed, the results obtained, and the conclusions drawn.
•	Facial.Features: This folder contains two Python files, Detect.Face.py and FeatureEng.py. The Detect.Face.py file includes the code for detecting faces and extracting facial attributes from a video, while the FeatureEng.py file includes the code for engineering facial features.
•	MovieSalePrediction.ipynb: This Python file consists of the code for predicting movie sales based on the facial features of actors. It has several machine learning models and the hyperparameter tuning part as well.
•	FeatureImportanceViaFeaturePermutation.py: This Python file includes the code for finding important features using feature permutation. It employs the HistGBR model for feature importance calculation.
Usage
To run the project, clone the repository and run MovieSalePrediction.py. This file includes a primary function that loads the pre-processed data, trains the machine-learning models, and evaluates their performance.
For feature importance calculation, run FeatureImportanceViaFeaturePermutation.py. This file reads the pre-processed data and employs the HistGRB model to calculate feature importance using permutation.
Note that the Facial.Features folder is not necessary for running the project as it contains the code for data gathering, which has already been performed and pre-processed.
Conclusion
This project demonstrates the potential of using trailer content, especially facial features for predicting movie sales. The machine learning models employed in the project achieved agood performance, and feature permutation analysis indicated the importance of several facial features in predicting movie sales.
We hope this project inspires further research into using facial features to predict consumer behavior.
