**Vijey Shrivathsan, V.Sri Harsha affliated to T.Senthil Kumar,Associate Professor,Computer Science and Engineeirng Department,Amrita Vishwa Vidyapeetham,Coimbatore,Email-t_senthilkumar@cb.amrita.edu, Sulakshan Vajipayajula-IBM. Code Developed as part of IBM Funded Project: Detection and Prevention of Advanced Persistent Threat (APT) activities in heterogeneous networks using SIEM and Deep Learning**

The organization consists of different networks at various geographical locations. For such vast networks a simple honeypot is not enough to decoy attackers. Hence, a collection of various honeypots installed at various geographically separated locations inside the organization is necessary for luring attackers. Such a conglomeration of honeypots – *Honeynet* – is the key in collection of attacker data and traffic destined at the organization. Heterogeneous data from Network devices, Systems, Firewalls, NIDS, UTMs, etc., are collected at a centralized location using *Cloud basedSplunk Security Information and Event Management (SIEM)* for further processing. Extracting useful information from a plethora of heterogeneous data is a difficult task. SIEM is supported with a *Correlation Engine* for processing such heterogeneous data. The Correlation Engine is capable of deploying Complex Event Analysis techniques, Data Mining techniques, Deep Learning algorithms, Log Analysis techniques, etc., for searching the presence of attack vectors (or anomalous behaviour). The output of the Correlation Engine can be categorised to rank the output network behaviour in terms of the severity of the data/traffic by using a metric such as *Vulnerability Score*. The dashboard of the SIEM machine is capable of displaying the near real time processing of the various network and host events, network traffic flow statistics, system behaviour, and other properties of the network.

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
