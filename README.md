# openscope-barcodingstim

## Dependencies:
* Windows OS (see Camstim package)
* python 2.7
* psychopy 1.82.01
* camstim 0.2.4  

## Camstim 0.2.4:
* Built and licensed by the Allen Institute.
* Written in Python 2 and designed for Windows OS (requires pywin32).
* Pickled stimulus presentation logs are typically saved under user/camstim/output. 

## Installation with Anaconda or Miniconda:
1. Navigate to repository and install conda environment.
`conda env create -f openscope-glo-stim.yml`
2. Activate the environment.
`conda activate openscope-glo-stim`
3. Install the AIBS camstim package in the environment.
`pip install camstim/.`
4. Download and install `AVbin` for your OS.
 
## Pilot script
* Run the pilot_script.py under test-scripts folder to run Full Field Flicker and Static Grating and Drifting Grating
* Currently it does not have Debug mode
* Stimulus files are concatenated in this script 
