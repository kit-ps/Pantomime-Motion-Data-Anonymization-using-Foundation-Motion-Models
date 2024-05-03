## Overview
This repository contains the point light display renders from the paper "Pantomime: Towards the Anonymization of Motion Data using Foundation Motion Models". The motion sequences are the those used as stimuli for the user study.

The files for the two datasets can be found in their respective folders. We provide a simple overview for each dataset showing the anonymizations of a single motion sequence side by side, [Horst Overview](Horst-Study/Horst-Study_overview.html) and [CeTI-Locomotion Overview](CeTI-Locomotion/CeTI-Locomotion_overview.html). In general, the combination of VPoser fitting and VPoser latent encoding for anonymization produces the best looking results. In the overview, the used sample from the datasets is written with the given recognition target, either 0.1 for 10% recognition accuracy or 0.2 for 20% recognition accuracy. By scaling the anonymizations to achieve the same recognition goal, their results can be directly compared.


## Examples

### A good example from Horst-DB (VPoser fit + VPoser enc.)
![](Horst-Study/Horst-Study_S43.0011-Gait.npz_vposer_fitting_vposer_0.1.gif).

### A good example from CeTI-Locomotion (VPoser fit + VPoser enc.)
![](CeTI-Locomotion/CeTI-Locomotion_sub-K60.task-gaitFast-tracksys-RokokoSmartSuitPro1-run-1-step-1-motion.npz_vposer_fitting_vposer_0.1.gif)


### A bad example from Horst-DB (HuMoR fit + HuMoR enc.)
![](Horst-Study/Horst-Study_S26.0006-Gait.npz_humor_fitting_humor_0.1.gif)

### A bad example from CeTI-Locomotion (HuMoR fit + VPoser enc.)
![](CeTI-Locomotion/CeTI-Locomotion_sub-K39.task-sts-tracksys-RokokoSmartSuitPro1-run-2-sequence-2-motion.npz_humor_fitting_vposer_0.1.gif)

