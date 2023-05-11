
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

# read Contrast information from comma separated file
def read_file(path):
    with open(path) as f:
        Contrast = f.readlines()
        Contrast = [x.split(',') for x in Contrast]
        Contrast = [x for x in Contrast if len(x) > 1]
        Contrast = [[y.strip() for y in x] for x in Contrast]
        # remove empty strings
        Contrast = [[y for y in x if y] for x in Contrast]
        Contrast = [[float(y) for y in x] for x in Contrast]
        return Contrast[0]

def create_flashes(list_of_contrasts, window, n_repeats, frame_rate, current_start_time):
    """Create flash stimulus series implemented as a grating with fixed 
    sf=0, ori=0, and ph=0. contrast is updated every video frame according to 
    the loaded stimulus sequence.
        args:
            list_of_contrasts: list of contrasts to be presented
            window: window object
            n_repeats: number of repeats of the stimulus sequence
            frame_rate: frame rate of the monitor
            current_start_time: current start time of the stimulus
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
                'Contrast': ([1], 0),
                'Color':(list_of_contrasts, 1)
                },
        sweep_length=1.0/frame_rate,
        start_time=0.0,
        blank_length=0,
        blank_sweeps=0,
        runs = n_repeats,
        shuffle=False,
        save_sweep_table=True,
        )
    
    # For flashes, the duration of the stimulus is the number of unique stimuli 
    # divided by the frame rate
    duration_stim = len(list_of_contrasts)/frame_rate 
    end_stim = current_start_time+duration_stim
    
    stimulus_obj.set_display_sequence([(current_start_time, end_stim)])

    return stimulus_obj, end_stim

def create_static(list_of_contrasts, window, n_repeats, frame_rate, current_start_time,
                  list_of_spatialfreq, list_of_orientations, list_of_phases):
    """Create static grating stimulus series.
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
                'SF': (list_of_spatialfreq, 0),
                'Ori': (list_of_orientations, 1),
                'Phase': (list_of_phases, 2),
                'Contrast': (list_of_contrasts, 3),
                },
        sweep_length=1.0/frame_rate,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=n_repeats,
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

    return stimulus_obj, end_stim

def create_drift(window, n_repeats, frame_rate, current_start_time,
                  list_of_spatialfreq, list_of_orientations, 
                  list_of_drifts, drift_rates):
    """Create drifting grating stimulus series.
        args:
            window: window object
            n_repeats: number of repeats of the stimulus sequence
            frame_rate: frame rate of the monitor
            current_start_time: current start time of the stimulus
            list_of_spatialfreq: list of spatial frequencies to be presented
            list_of_orientations: list of orientations to be presented
            list_of_drifts: list of drifts to be presented
            drift_rates: list of drift rates to be presented
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

    for drift_coefficient in drift_rates:
        for drift_direction in list_of_drifts:
            # we operate in degrees here (0-360)
            current_phase = current_phase + drift_coefficient*drift_direction/frame_rate
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
                'Contrast': ([1], 0),
                'SF': (list_of_spatialfreq, 1),
                'Ori': (list_of_orientations, 2),
                'Phase': (list_of_phases, 3),
        },
        sweep_length=1.0/frame_rate,
        start_time=0.0,
        blank_length=0,
        blank_sweeps=0,
        runs=n_repeats,
        shuffle=False,
        save_sweep_table=True,
        )

    # The duration of the stimulus is the number of unique stimuli 
    # divided by the frame rate
    number_conditions = len(list_of_spatialfreq)*\
        len(list_of_orientations)*len(list_of_phases)
    
    logging.info('Number of conditions for drifting gratings: %s', number_conditions)

    duration_stim = number_conditions*n_repeats/frame_rate
    end_stim = current_start_time+duration_stim
    
    stimulus_obj.set_display_sequence([(current_start_time, end_stim)])

    return stimulus_obj, end_stim

def get_stimulus_sequence(window, SESSION_PARAMS_data_folder):

    ################# Parameters #################
    FPS = 60
    SPATIALFREQ = [0.02, 0.04, 0.08]
    ORIENTATIONS = [0, 90]
    PHASES = [0, 90]
    DRIFTRATES = [12, 24]
    ADD_FLASHES = True
    ADD_STATIC = True
    ADD_DRIFT = True
    Nrepeats = 1 # 32 # number of time the repeated sequences repeat
    ##############################################

    # Read in the stimulus sequence
    # UniqueStim are not currently used in the pilot script
    # UniqueStim1 =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
    #                                           "UniqueStim1.txt")) 
    #      
    # UniqueStim2 =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
    #                                           "UniqueStim2.txt"))     


    RepeatStim1 =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
                                             "BinoWhiteNoise_8Sec.txt")) #NOTE CHANGE PR
    
    RepeatStim2 =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
                                              "ExpWhiteNoise_8sec.txt")) #NOTE CHANGE PR

    RepeatStim3 =  read_file(os.path.join(SESSION_PARAMS_data_folder, 
                                              "GaussWhiteNoise_8sec.txt")) #NOTE CHANGE PR

    # This is used to keep track of the current start time of the stimulus
    current_start_time = 0

    # This is a list of all the stimuli that will be presented
    all_stim = []

    if ADD_FLASHES:
        flash_sequence1, current_start_time = create_flashes(
                RepeatStim1, window, Nrepeats, FPS, current_start_time
                )
        all_stim.append(flash_sequence1)

        flash_sequence2, current_start_time = create_flashes(
                RepeatStim2, window, Nrepeats, FPS, current_start_time
                )
        all_stim.append(flash_sequence2)

        flash_sequence3, current_start_time = create_flashes(           
                RepeatStim3, window, Nrepeats, FPS, current_start_time
                )
        all_stim.append(flash_sequence3)
        logging.info("Flashes end at : %f min", current_start_time/60)

    if ADD_STATIC:
        sg_sequence, current_start_time = create_static(
                RepeatStim1, window, Nrepeats, FPS, current_start_time,
                SPATIALFREQ, ORIENTATIONS, PHASES
                )
        
        all_stim.append(sg_sequence)
        logging.info("Static gratings end at : %f min", current_start_time/60)

    if ADD_DRIFT:
        dg_sequence, current_start_time = create_drift(
                window, Nrepeats, FPS, current_start_time,
                SPATIALFREQ, ORIENTATIONS, RepeatStim1, DRIFTRATES
                )
        
        all_stim.append(dg_sequence)    
        logging.info("Drifting gratings end at : %f min", current_start_time/60)

    logging.info("Total duration of stimulus sequence: %f min", current_start_time/60)
    # This is the stimulus sequence that will be presented
    return all_stim


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mtrain")
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    
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

    sequence_stim = get_stimulus_sequence(window, SESSION_PARAMS_data_folder)

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

    # Run the SweepStim instance
    t = time.localtime()
    ss.run()
    logging.info("total time: %f", t - time.localtime())