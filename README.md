**V.Sri Harsha, Vijey Shrivathsan in collaboration with T.Senthil Kumar,Associate Professor,Computer Science and Engineering Department, Amrita Vishwa Vidyapeetham, Coimbatore, India, Email-t_senthilkumar@cb.amrita.edu and Sulakshan Vajipayajula, IBM. 
Code Developed as part of IBM Funded Project: Two-Stage IDS Using Deep Learning**

Network intrusion detection system (NIDS) is a tool used to detect and classify the network breaches dynamically in information and communication technologies (ICT) systems in both industries and academia. NIDS is used to detect network born attacks such as Denial of Service (DoS) attacks, malware replication, and intruders that are operating within the system.
Deep learning algorithms and frameworks have revolutionized predictive analysis over the past decade. These powerful techniques can be leveraged in the field of Intrusion Detection to classify and predict cyber-attacks with minimal overhead. The dynamic nature of the problem along with the arise of new network attacks, make this problem highly intricate. In this project, we explore *LSTM-Autoencoders* and a unique *two-stage deep learning framework* for NIDS. The work is done on the *CICIDS-17* dataset which is a comprehensive dataset with an amalgam of real, modern, normal and contemporary attacks. We propose this deep neural network to classify the attacks using flow-based traffic with a significant classification accuracy higher than that of existing deep learning frameworks. 
# Installation:

#### 1. Download the dataset from

    i. https://www.unb.ca/cic/datasets/ids-2017.html (or)
    ii. https://drive.google.com/open?id=1CARwQLIgqNcxObqOoGey-3TzaShq9tHh 
#### 2. Prepare Data Files
Copy all the csv files from the "MachineLearningCVE" folder and paste them in "FlowBasedNIDS" folder.

# Running:
1. Execute LstmAEModelGenerator.py to generate "lstm_ae.h5", a model that can classify traffic as attack or benign.
2. Execute ANNModelGenerator.py to generate "attacks.h5", a model that can predict the class of an attack.
3. Execute TwoStageClassifier.py that takes the generated models as input and and predicts the traffic on the test data.
