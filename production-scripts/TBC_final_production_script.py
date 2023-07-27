# Modified by P.Reinagel 230726
import camstim
from camstim import Stimulus, SweepStim, Foraging, Window, Warp
from psychopy import monitors, visual
import os
import time
import numpy as np
import logging
import sys
import argparse
import yaml

"""
runs white noise stimuli for temporal barcoding project 
   by preinagel@ucsd.edu
runs optotagging code for ecephys pipeline experiments
   by joshs@alleninstitute.org, corbettb@alleninstitute.org, chrism@alleninstitute.org, jeromel@alleninstitute.org

(c) 2018 Allen Institute for Brain Science
"""
import datetime
import csv
import pickle as pkl
from copy import deepcopy

def run_optotagging(levels, conditions, waveforms, isis, sampleRate = 10000.):

    from toolbox.IO.nidaq import AnalogOutput
    from toolbox.IO.nidaq import DigitalOutput

    sweep_on = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    stim_on = np.array([0,0,1,1,0,0,0,0], dtype=np.uint8)
    stim_off = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    sweep_off = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)

    ao = AnalogOutput('Dev1', channels=[1])
    ao.cfg_sample_clock(sampleRate)

    do = DigitalOutput('Dev1', 2)

    do.start()
    ao.start()

    do.write(sweep_on)
    time.sleep(5)

    for i, level in enumerate(levels):

        print(level)

        data = waveforms[conditions[i]]

        do.write(stim_on)
        ao.write(data * level)
        do.write(stim_off)
        time.sleep(isis[i])

    do.write(sweep_off)
    do.clear()
    ao.clear()

def generatePulseTrain(pulseWidth, pulseInterval, numRepeats, riseTime, sampleRate = 10000.):

    data = np.zeros((int(sampleRate),), dtype=np.float64)
   # rise_samples =

    rise_and_fall = (((1 - np.cos(np.arange(sampleRate*riseTime/1000., dtype=np.float64)*2*np.pi/10))+1)-1)/2
    half_length = int(rise_and_fall.size / 2)
    rise = rise_and_fall[:half_length]
    fall = rise_and_fall[half_length:]

    peak_samples = int(sampleRate*(pulseWidth-riseTime*2)/1000)
    peak = np.ones((peak_samples,))

    pulse = np.concatenate((rise, \
                           peak, \
                           fall))

    interval = int(pulseInterval*sampleRate/1000.)

    for i in range(0, numRepeats):
        data[i*interval:i*interval+pulse.size] = pulse

    return data

def optotagging(mouse_id, operation_mode='experiment', level_list = [1.15, 1.28, 1.345], output_dir = 'C:/ProgramData/camstim/output/'):

    sampleRate = 10000

    # 1 s cosine ramp:
    data_cosine = (((1 - np.cos(np.arange(sampleRate, dtype=np.float64)
                                * 2*np.pi/sampleRate)) + 1) - 1)/2  # create raised cosine waveform

    # 1 ms cosine ramp:
    rise_and_fall = (
        ((1 - np.cos(np.arange(sampleRate*0.001, dtype=np.float64)*2*np.pi/10))+1)-1)/2
    half_length = int(rise_and_fall.size / 2)

    # pulses with cosine ramp:
    pulse_2ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.001),)), rise_and_fall[half_length:]))
    pulse_5ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.004),)), rise_and_fall[half_length:]))
    pulse_10ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.009),)), rise_and_fall[half_length:]))

    data_2ms_10Hz = np.zeros((sampleRate,), dtype=np.float64)

    for i in range(0, 10):
        interval = int(sampleRate / 10)
        data_2ms_10Hz[i*interval:i*interval+pulse_2ms.size] = pulse_2ms

    data_5ms = np.zeros((sampleRate,), dtype=np.float64)
    data_5ms[:pulse_5ms.size] = pulse_5ms

    data_10ms = np.zeros((sampleRate,), dtype=np.float64)
    data_10ms[:pulse_10ms.size] = pulse_10ms

    data_10s = np.zeros((sampleRate*10,), dtype=np.float64)
    data_10s[:-2] = 1

    ##### THESE STIMULI ADDED FOR OPENSCOPE GLO PROJECT #####
    data_10ms_5Hz = generatePulseTrain(10, 200, 5, 1) # 1 second of 5Hz pulse train. Each pulse is 10 ms wide
    data_6ms_40Hz = generatePulseTrain(6, 25, 40, 1)  # 1 second of 40 Hz pulse train. Each pulse is 6 ms wide
    #########################################################

    # for experiment

    isi = 1.5
    isi_rand = 0.5
    numRepeats = 50

    condition_list = [3, 4, 5]
    waveforms = [data_2ms_10Hz, data_5ms, data_10ms, data_cosine, data_10ms_5Hz, data_6ms_40Hz]

    opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
    opto_conditions = condition_list*numRepeats*len(level_list)
    opto_conditions = np.sort(opto_conditions)
    opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi

    p = np.random.permutation(len(opto_levels))

    # implement shuffle?
    opto_levels = opto_levels[p]
    opto_conditions = opto_conditions[p]

    # for testing

    if operation_mode=='test_levels':
        isi = 2.0
        isi_rand = 0.0

        numRepeats = 2

        condition_list = [0]
        waveforms = [data_10s, data_10s]

        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi

    elif operation_mode=='pretest':
        numRepeats = 1

        condition_list = [0]
        data_2s = data_10s[-sampleRate*2:]
        waveforms = [data_2s]

        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = [1]*len(opto_conditions)
    #

    outputDirectory = output_dir
    fileDate = str(datetime.datetime.now()).replace(':', '').replace(
        '.', '').replace('-', '').replace(' ', '')[2:14]
    fileName = os.path.join(outputDirectory, fileDate + '_'+mouse_id + '.opto.pkl')

    print('saving info to: ' + fileName)
    fl = open(fileName, 'wb')
    output = {}

    output['opto_levels'] = opto_levels
    output['opto_conditions'] = opto_conditions
    output['opto_ISIs'] = opto_isis
    output['opto_waveforms'] = waveforms

    pkl.dump(output, fl)
    fl.close()
    print('saved.')

    #
    run_optotagging(opto_levels, opto_conditions,
                    waveforms, opto_isis, float(sampleRate))
"""
end of optotagging section
"""

def read_file(path):
    # read Contrast information from comma separated file
    # modified by PR 230726 to simplify
    with open(path) as f:
        csvreader=csv.reader(f)
        Contrast=list(csvreader)[0]
        # Remove empty cell and cast to float
        Contrast = [float(x) for x in Contrast if x != '']
    return Contrast

def create_fullfieldflicker(list_of_contrasts, window, n_repeats, frame_rate, current_start_time, stimname):
    # modified by PR 230726 to add stimname argument
    """Create full field flicker stimulus implemented as a grating with fixed 
    sf=0, ori=0, and ph=0 whose contrast is updated every video frame according to 
    the loaded stimulus sequence.
        args:
            list_of_contrasts: list of contrasts to be presented
            window: window object
            n_repeats: number of repeats of the stimulus sequence
            frame_rate: frame rate of the monitor
            current_start_time: current start time of the stimulus
            stimname: name for stimulus table
        returns:
            stimulus_obj: CamStim Stimulus object
            end_stim: updated current start time of the next stimulus
    """
    stimulus_obj = Stimulus(visual.GratingStim(window,
                        pos=(0, 0),
                        units='deg',
                        size=(250, 250),
                        mask="None",
                        texRes=256,
                        sf=0,
                        ),
        # a dictionary that specifies the parameter values
        # that will be swept or varied over time during the presentation of the stimulus
        sweep_params={
                    # works similarly like for loops
                    # for a fixed contrast value, Contrast is updated every video frame
                'index_repeat': (np.arange(n_repeats), 0),
                'Contrast': ([1], 1),
                'Color':(list_of_contrasts, 2)
                },
        sweep_length=1.0/frame_rate,
        start_time=0.0,
        blank_length=0,
        blank_sweeps=0,
        runs = 1,
        shuffle=False,
        save_sweep_table=True,
        )
    
    # For spatially uniform white noise (full field flicker) stimuli, the duration of the 
    # stimulus is the number of unique stimulus values provided divided by the frame rate
    duration_stim = n_repeats*len(list_of_contrasts)/frame_rate 
    end_stim = current_start_time+duration_stim
    
    stimulus_obj.set_display_sequence([(current_start_time, end_stim)])
    stimulus_obj.stim_path = r"C:\\not_a_stim_script\\"+stimname+".stim"  

    return stimulus_obj, end_stim

def create_static(list_of_contrasts, window, n_repeats, frame_rate, current_start_time,
                  list_of_spatialfreq, list_of_orientations, list_of_phases):
    """Create static grating stimulus that changes contrast every frame according to loaded timeseries
        args:
            list_of_contrasts: list of contrasts to be presented
            window: window object
            n_repeats: number of repeats of the stimulus sequence
            frame_rate: frame rate of the monitor
            current_start_time: current start time of the stimulus
            list_of_spatialfreq: list of spatial frequencies to be presented
            list_of_orientations: list of orientations to be presented
            list_of_phases: list of phases to be presented
        returns:
            stimulus_obj: CamStim Stimulus object
            end_stim: updated current start time of the next stimulus
    """

    # Standing (static) Grating with fixed sf, ori, and ph
    # contrast is updated every video frame according to the loaded stimulus sequence
    stimulus_obj = Stimulus(visual.GratingStim(window,
                        pos=(0, 0),
                        units='deg',
                        tex="sin",
                        size=(250, 250),
                        mask="None",
                        texRes=256,
                        sf=0.1, # this mimmicks visual coding experiments
                        ),
        # a dictionary that specifies the parameter values
        # that will be swept or varied over time during the presentation of the stimulus
        sweep_params={
                    # wokrs similarly like for loops
                    # for a fixed contrast value, Contrast is updated every video frame
                'index_repeat': (np.arange(n_repeats), 0),
                'SF': (list_of_spatialfreq, 1),
                'Ori': (list_of_orientations, 2),
                'Phase': (list_of_phases, 3),
                'Contrast': (list_of_contrasts, 4),
                },
        sweep_length=1.0/frame_rate,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=1,
        shuffle=False,
        save_sweep_table=True,
        )

    # The duration of the stimulus is the number of unique stimuli 
    # divided by the frame rate, where each frame of the contrast time series is
    # considered a separate stimulus by camstim
    number_conditions = len(list_of_spatialfreq)*\
        len(list_of_phases)*len(list_of_orientations)*len(list_of_contrasts)

    logging.info('Number of conditions for static gratings: %d', number_conditions)

    duration_stim = number_conditions*n_repeats/frame_rate
    end_stim = current_start_time+duration_stim
    
    stimulus_obj.set_display_sequence([(current_start_time, end_stim)])
    stimulus_obj.stim_path = r"C:\\not_a_stim_script\\static_block.stim"

    return stimulus_obj, end_stim

def create_drift(window, n_repeats, frame_rate, current_start_time,
                  list_of_spatialfreq, list_of_orientations, 
                  list_of_drifts, drift_rate):
    """Create drifting grating stimulus series.
        args:
            window: window object
            n_repeats: number of repeats of the stimulus sequence
            frame_rate: frame rate of the monitor
            current_start_time: current start time of the stimulus
            list_of_spatialfreq: list of spatial frequencies to be presented
            list_of_orientations: list of orientations to be presented
            list_of_drifts: list of drifts to be presented
            drift_rate: drift rate to be used
        returns:
            stimulus_obj: CamStim Stimulus object
            end_stim: updated current start time of the next stimulus
    """

    # a standing square-wave grating (black and white stripes with sharp edges) 
    # with a specified spatial frequency and orientation, which starts with a 
    # phase of 0 and then drifts in the direction orthogonal to the stripes 
    # updating the drift rate (edge speed) every video frame according to a 
    # sequence that is passed in as a vector of floating-point values ranging 
    # from -1 to 1, where for vertical stripes -1 is the maximum speed in 
    # leftward direction and +1 is maximum speed in rightward direction.
    list_of_phases = []

    current_phase = 0.0

    for drift_direction in list_of_drifts:
        # we operate in degrees here (0-360)
        current_phase = current_phase + drift_rate*drift_direction/frame_rate
        current_phase = np.mod(current_phase, 360)
        list_of_phases.append(current_phase)

    # psychopy is unconventional in that phases have modulus 1
    list_of_phases = [x/360 for x in list_of_phases]

    stimulus_obj = Stimulus(visual.GratingStim(window,
                        pos=(0, 0),
                        units='deg',
                        tex="sqr",
                        size=(250, 250),
                        mask="None",
                        texRes=256,
                        sf=0.1, # this mimmicks visual coding experiments
                        ),
        sweep_params={
                'index_repeat': (np.arange(n_repeats), 0),
                'Contrast': ([1], 1),
                'SF': (list_of_spatialfreq, 2),
                'Ori': (list_of_orientations, 3),
                'Phase': (list_of_phases, 4),
        },
        sweep_length=1.0/frame_rate,
        start_time=0,
        blank_length=0,
        blank_sweeps=0,
        runs=1,
        shuffle=False,
        save_sweep_table=True,
        )
    stimulus_obj.stim_path = r"C:\\not_a_stim_script\\drift_block_driftrate_"+str(drift_rate)+".stim"

    # The duration of the stimulus is the number of unique stimuli 
    # divided by the frame rate
    number_conditions = len(list_of_spatialfreq)*\
        len(list_of_orientations)*len(list_of_phases)
    
    logging.info('Number of conditions for drifting gratings: %s', number_conditions)

    duration_stim = number_conditions*n_repeats/frame_rate
    end_stim = current_start_time+duration_stim
    
    stimulus_obj.set_display_sequence([(current_start_time, end_stim)])

    return stimulus_obj, end_stim

def create_receptive_field_mapping(window, number_runs = 15):
    x = np.arange(-40,45,10)
    y = np.arange(-40,45,10)
    position = []
    for i in x:
        for j in y:
            position.append([i,j])

    stimulus = Stimulus(visual.GratingStim(window,
                        units='deg',
                        size=20,
                        mask="circle",
                        texRes=256,
                        sf=0.1,
                        ),
        sweep_params={
                'Pos':(position, 0),
                'Contrast': ([0.8], 4),
                'TF': ([4.0], 1),
                'SF': ([0.08], 2),
                'Ori': ([0,45,90], 3),
                },
        sweep_length=0.25,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=number_runs,
        shuffle=True,
        save_sweep_table=True,
        )
    stimulus.stim_path = r"C:\\not_a_stim_script\\receptive_field_block.stim"

    return stimulus

def get_stimulus_sequence(window, SESSION_PARAMS_data_folder, ADD_FULLFIELD, ADD_STATIC, ADD_DRIFT, ADD_RF, Nrepeats, number_runs_rf):

    ################# Parameters #################
    # PR will be changing these parameters - TBD
    FPS = 60            
    SPATIALFREQ = [0.02, 0.04, 0.08] 
    ORIENTATIONS = [0, 90]
    PHASES = [0.0, 90.0/360]  
    DRIFTRATES = [360, 540]  
    ##############################################

    # Read in the stimulus sequences from csv files
    WhiteNoiseLong =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
                                             "TBC_WhiteNoise_120sec.txt")) # NEW FILE WILL BE PROVIDED

    WhiteNoiseShort =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
                                              "TBC_WhiteNoise_8sec.txt"))  # NEW FILE WILL BE PROVIDED

    # This is used to keep track of the current start time of the stimulus
    current_start_time = 0

    # This is a list of all the stimuli that will be presented
    all_stim = []

    if ADD_UNIQUE:
        # Create stimulus that plays one long unique white noise sequence
        norepeats=1 #override Nrepeats for this stimulus only
        fullfieldWN_unique, current_start_time = create_fullfieldflicker(
                WhiteNoiseLong, window, norepeats, FPS, current_start_time, # <-- N.B. norepeats
                'UniqueFFF')
        all_stim.append(fullfieldWN_unique)
        logging.info("Unique white noise FFF ends at : %f min", current_start_time/60)

    if ADD_FULLFIELD:
        # Create stimulus that plays short white noise sequence 2*Nrepeats times <- N.B. 2*Nrepeats
        fullfieldWN_repeated, current_start_time = create_fullfieldflicker(           
                WhiteNoiseShort, window, 2*Nrepeats, FPS, current_start_time,
                'RepeatFFF')
        all_stim.append(fullfieldWN_repeated)
        logging.info("Repeated white noise FFF ends at : %f min", current_start_time/60)

    if ADD_UNIQUE: #intentionally duplicated we want this module repeated at this time
        # Create stimulus that plays one long unique white noise sequence
        norepeats=1 #override Nrepeats for this stimulus only
        fullfieldWN_unique, current_start_time = create_fullfieldflicker(
                WhiteNoiseLong, window, norepeats, FPS, current_start_time, # <-- N.B. norepeats
                'UniqueFFF')
        all_stim.append(fullfieldWN_unique)
        logging.info("Second unique white noise FFF ends at : %f min", current_start_time/60)

    if ADD_STATIC:
        sg_sequence, current_start_time = create_static(
                WhiteNoiseShort, window, Nrepeats, FPS, current_start_time,
                SPATIALFREQ, ORIENTATIONS, PHASES
                )
        
        all_stim.append(sg_sequence)
        logging.info("Static gratings end at : %f min", current_start_time/60)

    if ADD_DRIFT:
        for drift_rate in DRIFTRATES:
            dg_sequence, current_start_time = create_drift(
                    window, Nrepeats, FPS, current_start_time,
                    SPATIALFREQ, ORIENTATIONS, WhiteNoiseShort, drift_rate
                    )

            all_stim.append(dg_sequence)    
            logging.info("Drifting gratings end at : %f min", current_start_time/60)

    if ADD_RF:
        # 1 minute per repeat, 8 repeats preferred
        gabors_rf_20  = create_receptive_field_mapping(window, number_runs_rf)
        gabors_rf_20_ds = [(current_start_time, current_start_time+60*number_runs_rf)]
        gabors_rf_20.set_display_sequence(gabors_rf_20_ds)

        all_stim.append(gabors_rf_20)    
        logging.info("Receptive fields end at : %f min", (current_start_time+60*number_runs_rf)/60)

    return all_stim


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mtrain")
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    logging.basicConfig(level=logging.INFO)
    # logging.info args
    if args.json_path == "":
        logging.warning("No json path provided, using default parameters. THIS IS NOT THE EXPECTED BEHAVIOR FOR PRODUCTION RUNS")
        json_params = {}
    else:
        with open(args.json_path, 'r') as f:
            # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
            json_params = yaml.load(f)
            logging.info("Loaded json parameters from mtrain")
            # end of mtrain part

    dist = 15.0
    wid = 52.0

    # mtrain should be providing : a path to a network folder or a local folder with the entire repo pulled
    SESSION_PARAMS_data_folder = json_params.get('data_folder', os.path.dirname(os.path.abspath(__file__)))

    # mtrain should be providing : Gamma1.Luminance50
    monitor_name = json_params.get('monitor_name', "testMonitor")

    # mtrain should be providing : 
    ADD_UNIQUE = json_params.get('add_unique', True)
    ADD_FULLFIELD = json_params.get('add_fullfield', True)
    ADD_STATIC = json_params.get('add_static', True)
    ADD_DRIFT = json_params.get('add_drift', True)
    ADD_RF = json_params.get('add_rf', True)
    number_runs_rf = json_params.get('number_runs_rf', 1) # 8 is the number of repeats for prod(8min).
    Nrepeats = json_params.get('n_repeats', 1) # 
    opto_disabled = json_params.get('disable_opto', True)

    # create a monitor
    if monitor_name == 'testMonitor':
        monitor = monitors.Monitor(monitor_name, distance=dist, width=wid)
    else:
        monitor = monitor_name
        
    # Create display window
    window = Window(fullscr=True, # Will return an error due to default size. Ignore.
                    monitor=monitor,  # Will be set to a gamma calibrated profile by MPE
                    screen=0,
                    warp=Warp.Spherical
                    )

    sequence_stim = get_stimulus_sequence(window, SESSION_PARAMS_data_folder,
                                          ADD_UNIQUE = ADD_UNIQUE,     
                                          ADD_FULLFIELD = ADD_FULLFIELD,
                                        ADD_STATIC = ADD_STATIC,
                                        ADD_DRIFT = ADD_DRIFT,
                                        ADD_RF = ADD_RF,
                                        Nrepeats = Nrepeats,
                                        number_runs_rf = number_runs_rf
                                        )      

    ss = SweepStim(window,
                    stimuli=sequence_stim,
                    pre_blank_sec=0,
                    post_blank_sec=0,
                    params={},
                    )

    # add in foraging so we can track wheel, potentially give rewards, etc
    f = Foraging(window = window,
                    auto_update = False,
                    params= {}
                    )
    
    ss.add_item(f, "foraging")

     # run it
    try:
        ss.run()
    except SystemExit:
        print("We prevent camstim exiting the script to complete optotagging")

    if not(opto_disabled):
        from camstim.misc import get_config
        from camstim.zro import agent
        opto_params = deepcopy(json_params.get("opto_params"))
        opto_params["mouse_id"] = json_params["mouse_id"]
        opto_params["output_dir"] = agent.OUTPUT_DIR
        #Read opto levels from stim.cfg file
        config_path = agent.CAMSTIM_CONFIG_PATH
        stim_cfg_opto_params = get_config(
            'Optogenetics',
            path=config_path,
        )
        opto_params["level_list"] = stim_cfg_opto_params["level_list"]

        optotagging(**opto_params)