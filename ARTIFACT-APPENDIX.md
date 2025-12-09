# Artifact Appendix (Required for all badges)

Paper title: **Pantomime: Motion Data Anonymization using Foundation Motion Models**

Requested Badge(s):
  - [x] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**

## Description
This repository contains the code for the paper "Pantomime: Motion Data 
Anonymization using Foundation Motion Models". Pantomime anonymizes 3D motion
data using HuMoR and VPoser motion models. These models require that the motion
data be encoded as SMPL parameters. If the motion data is not available in the
SMPL format, an additional fitting step is required to convert it. Please note
that this process can be time-consuming. In addition to the anonymization code,
this repository contains the evaluation code used for identification experiments.

In addition to the code necessary to run Pantomime, this repository contains an 
overview of motion sequences that have been anonymized using Pantomime, as well 
as different parameters.


### Security/Privacy Issues and Ethical Concerns (Required for all badges)
The code can be executed by an unprivileged user and does not require internet 
access. Both the CeTI-Locomotion and the Horst-DB datasets used in this study 
were approved by their respective ethics committees, and the participants gave 
written informed consent to participate in the data collection.


## Environment (Required for all badges)


### Accessibility (Required for all badges)
We provide a github repository:

https://github.com/kit-ps/Pantomime-Motion-Data-Anonymization-using-Foundation-Motion-Models/tree/main


## Notes on Reusability (Encouraged for all badges)


The code from Pantomime can be used to compare future motion anonymizations. 
The anonymization and biometric recognition codes can both be reused for this 
purpose. Furthermore, Pantomime's code can be expanded to test new foundation 
motion models and approaches for anonymization.
