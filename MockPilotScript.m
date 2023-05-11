% Pilot Script Mockup
% PR 230421
% NOTE ON DATA OUTPUTS
% Please put a timestamp in the data file at the beginning and end of each
% time a stimulus vector is displayed. If possible, also save the scalar
% stimulus value each video frame, or at least save the stimulus vector
% every time is played, for foolproof unambiguous reconstruction

clear,clc, close all
% parameters that may be changed by user
debugmode=true;
testmode=false;%true; % the fastest possible sweep through all stimuli
SessionDuration=120; % minutes for entire script if not in testmode
TimeDilation = 1; % consecutive video frames each stimulus value is displayed
% this is useful to slow down rapid stimuli to visually verify properties

%for pilot I will try binary white noise or an alternative distribution
fnameUnique1='./UniqueStim1.txt';% a file I will provide, 120s duration at 30fps
fnameRepeat1='./RepeatStim1.txt'; % a file I will provide, 8s duration at 30fps
fnameUnique2='./UniqueStim2.txt';% a file I will provide, 120s duration at 30fps
fnameRepeat2='./RepeatStim2.txt'; % a file I will provide, 8s duration at 30fps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I would bundle this section of code in a "setup" function within the
% file, so it's out of the way; but not sure if you can do that in python?

% Hardwired parameters
FPS=30; % A hardware framerate of 30FPS is assumed
FULLSESSION=120;%minutes duration of full length recording session
MAXRPT = 32;%#times to repeat the repeated stim, in full length sessions
% NOTE this will be ~400 in the production script
SPATIALFREQ = [0.02, 0.04, 0.08];% in cycles per degree
% NOTE one spatial frequency will likely be used in production
ORIENTATIONS = [0 90]; %in degrees
% NOTE 4 values will be used in production [0 45 90 135]
PHASES = [0 90];%spatial phase offset of grating in degrees, for standing
% NOTE 4 values will be used in production [0 45 90 135]
DRIFTRATES = [12 24]; %degrees of phase to move each frame, for drifting
% NOTE one step size will likely be used in production
NCOND = length(SPATIALFREQ)*length(PHASES)*length(ORIENTATIONS);

% Calculated Parameters
if testmode
    Nrepeats =1;% number of time the repeated sequences repeat
else
    Nrepeats = max([1, round(MAXRPT*SessionDuration/(120*TimeDilation))]);
end

% The stimulus files will contain comma separated lists of values
% ranging from -1 to 1, one value per video frame by default
% For debugging purposes I've just created random vectors, but for the
% pilot test on the mouse we will load the sequences from files
if debugmode
    BaseUniqueStim = binornd(1,0.5, 1,120*FPS)*2-1;
    BaseRepeatStim = binornd(1,0.5, 1,8*FPS)*2-1;
    BaseUniqueStim2 = exprnd(.5,1,120*FPS)-1;
    BaseUniqueStim2(BaseUniqueStim2>1)=1;
    BaseRepeatStim2 = exprnd(.5,1,8*FPS)-1;
    BaseRepeatStim2(BaseRepeatStim2>1)=1;
else
    BaseUniqueStim =  load(fnameUnique1);
    BaseRepeatStim =  load(fnameRepeat1);
    BaseUniqueStim2 =  load(fnameUnique2);
    BaseRepeatStim2 =  load(fnameRepeat2);
end
% Apply time dilation to the vectors
UniqueStim=[];UniqueStim2=[];%initialize
for i=1:length(BaseUniqueStim) %take each value in base vector
    for j=1:TimeDilation %and repeat it this many times in stimulus vector
        UniqueStim=[UniqueStim BaseUniqueStim(i)];
        UniqueStim2=[UniqueStim2 BaseUniqueStim2(i)];
    end
end
RepeatStim=[];RepeatStim2=[];%initialize
for i=1:length(BaseRepeatStim) %take each value in base vector
    for j=1:TimeDilation %and repeat it this many times in stimulus vector
        RepeatStim=[RepeatStim BaseRepeatStim(i)];
        RepeatStim2=[RepeatStim2 BaseRepeatStim(i)];
    end
end

DurationFFF = ... %duration in MINUTES of each FFF segment
    (2*length(UniqueStim)/FPS + Nrepeats*length(RepeatStim)/FPS) / 60;
DurationGR = ... %duration in MINUTES of each grating segment
    (NCOND*Nrepeats*length(RepeatStim)/FPS) / 60;
TotalScriptDuration=2*(DurationFFF+DurationGR); % in minutes
% Save the final stimulus vectors in the pkl file,
% along with the stimulus file names and all the constants & parameters
% we have set or calculated up to here

% SETUP FUNCTION ENDS HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT FOR THE EXPERIMENT BEGINS HERE
% PART 1 Full Field Flicker (FFF) with binary values
% where -1 encodes black, +1 encodes white
displayFFF(UniqueStim) % once through this stimulus = 2 min
for i=1:Nrepeats
    displayFFF(RepeatStim); % each repeat = 8 sec
end
displayFFF(UniqueStim); % = 2 min

% FFF stimulus with continuous floating point values from -1 to 1,
% encoding luminance values ranging smoothly from black to white
displayFFF(UniqueStim2)
for i=1:Nrepeats
    displayFFF(RepeatStim2);
end
displayFFF(UniqueStim2);

% PART 2 STANDING GRATINGS
% Loop over grating parameters, and then within a fixed parameter
% condition, modulate contrast according to the stimulus vector, one value
% per video frame; -1 encodes -100% contrast, +1 encodes +100% contrast
contrast=RepeatStim;
for i=1:length(SPATIALFREQ)
    sf = SPATIALFREQ(i);
    for j=1:length(ORIENTATIONS)
        ori = ORIENTATIONS(j);
        for k=1:length(PHASES)
            ph =  PHASES(k);
            % SAVE TO THE DATA FILE VALUES OF SF, ORI,PH at this timestamp
            % display the 8s repeated stimulus Nrepeats times
            for i=1:Nrepeats
                % contrast is a vector, one value per frame
                % sf, ori and ph are scalars
                displayFlickerGrating(contrast, sf, ori, ph);
            end
        end
    end
end

% PART 3 DRIFTING GRATINGS
% Loop over grating parameters and then within parameter condition,
% modulate contrast according to values in stimulus, one per video frame
% where -1 encodes changing the grating phase by -dr degrees, and +1
% encodes changing the phase by +dr degrees, compared to the phase of
% previous video frame. At 30fps dr=12 corresponds to 1Hz temporal
% frequency drift if it were shifting in a constant direction.
% Thus instead of saving the contrast of each frame to the data file, we
% should save the spatial phase of each frame

contrast=10;% contrast is fixed; this approximates a square wave grating?
for i=1:length(SPATIALFREQ)
    sf=SPATIALFREQ(i);
    for j=1:length(ORIENTATIONS)
        ori=ORIENTATIONS(j);
        for k=1:length(DRIFTRATES)
            dr=  DRIFTRATES(k);
            % SAVE TO THE DATA FILE VALUES OF SF, ORI, contrast, and DR of block
            
            % The following line of code starts from ph=0 and *changes* the
            % phase by RepeatStim(t)*dr every video frame; the MATLAB
            % function wrapTo360 then maps values below 0 or above 360
            % degrees to their equivalent in the 360 degree circle.
            ph=wrapTo360(cumsum(RepeatStim*dr));
            % display the 8s repeated stimulus Nrepeats times
            for i=1:Nrepeats
                % Contrast is a constant, ph is a vector one value per frame
                displayDriftingGrating(contrast, sf, ori, ph);
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT TEST FILES:
% fnames={'./UniqueStim1.txt','./RepeatStim1.txt',...
%     './UniqueStim2.txt','./RepeatStim2.txt'};
% varnames={'BaseUniqueStim', 'BaseRepeatStim'....
%     'BaseUniqueStim2','BaseRepeatStim2'};
% 
% for f=1:length(fnames) 
%     tmp=sprintf('X=%s;',varnames{f});eval(tmp) %put the variable in X
%     fid=fopen(fnames{f},'w');
%     for i=1:length(X),fprintf(fid,'%.5f,',X(i));end
%     fclose(fid);
% end
    


