
# Installation:

## 1. Download the dataset from

    i. https://www.unb.ca/cic/datasets/ids-2017.html (or)
    ii. https://drive.google.com/open?id=1CARwQLIgqNcxObqOoGey-3TzaShq9tHh 
## 2. Prepare Data Files
Copy all the csv files from the "MachineLearningCVE" folder and paste them in "FlowBasedNIDS" folder.

# Running:
1. Execute LstmAEModelGenerator.py to generate "lstm_ae.h5", a model that can classify traffic as attack or benign.
2. Execute ANNModelGenerator.py to generate "attacks.h5", a model that can predict the class of an attack.
3. Execute TwoStageClassifier.py that takes the generated models as input and and predicts the traffic on the test data.
