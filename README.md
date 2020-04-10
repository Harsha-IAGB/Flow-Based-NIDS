# Flow-Based-NIDS

Execution Flow:
1. Running LstmAEModelGenerator.py generates "lstm_ae.h5", a model that can classify traffic as attack or benign.
2. Running ANNModelGenerator.py generates "attacks.h5", a model that can predict the class of an attack.
3. Running TwoStageClassifier.py takes the generated models as input and and predicts the traffic on the test data.

Instructions:

1.The dataset can be downloaded from,

  i. https://www.unb.ca/cic/datasets/ids-2017.html
  
  ii. https://drive.google.com/open?id=1CARwQLIgqNcxObqOoGey-3TzaShq9tHh
  
2. Copy all the files from the "MachineLearningCVE" folder and paste them in "FlowBasedNIDS" folder.

3. Run the ModelGenerator files and then run the classifier.
