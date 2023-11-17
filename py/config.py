
import os
import numpy as np
import glob
# Fixed random seed
def getRandomState():

    RANDOM_SEED = 1024
    RANDOM = np.random.RandomState(RANDOM_SEED)

    return RANDOM

########################  DATASET  ########################





CLASS=['Cat', 'Chirpingbirds', 'Clapping', 'Coughing', 'Dog', 'Drinking', 'Drum', 'Female_speaking', 'Flute', 'Guitar', 'Hen', 'Hi-hat', 'Keyboard_typing', 'Kissing', 'Laughing', 'Manspeaking', 'Mike_disturbance', 'Mouse_click', 'Rooster', 'Silence', 'Sneezing', 'Snooring', 'Toilet_Flush', 'Tooth_brush', 'Vaccum_cleaner', 'Walk_Footsteps', 'Water', 'siren', 'whistling']
WHITE_LIST=['Clapping', 'Coughing','Female_speaking','Walk_Footsteps',  'whistling','Mike_disturbance', 'Mouse_click',  'Silence', 'Keyboard_typing']
######################  SPECTROGRAMS  ######################

# Type of frequency scaling, mel-scale = 'melspec', linear scale = 'linear'
SPEC_TYPE = 'melspec'
AMPLITUDE_SCALE = 'nonlinear'

# Sample rate for recordings, other sampling rates will force re-sampling
SAMPLE_RATE = 44100
RUN_NAME='AAL'
# Specify min and max frequency for low and high pass
SPEC_FMIN = 250
SPEC_FMAX = 25000

# Define length of chunks for spec generation, overlap of chunks and chunk min length
SPEC_LENGTH = 3.0
SPEC_OVERLAP = 0.0
SPEC_MINLEN = 1.0

#Threshold for distinction between noise and signal
SPEC_SIGNAL_THRESHOLD = 0.01

# Limit the amount of files and specs per class when extracting spectrograms (None = no limit)
MAX_FILES_PER_CLASS = 1000
MAX_SPECS_PER_CLASS = 10000

#########################  IMAGE  #########################
LOAD_OUTPUT_LAYER = True
# Number of channels
IM_DIM = 1

# Image size (width, height), should be the same as spectrogram shape
IM_SIZE = (512, 256)#PI: (320, 80)

NONLINEARITY='relu'

FILTERS=[16,32,64,128,256]

KERNEL_SIZES=[(7,7),(5,5),(3,3),(3,3),(3,3)]
NUM_OF_GROUPS=[1,1,1,1,1]
BATCH_NORM=True

DROPOUT_TYPE='channels'

DROPOUT=0.5

# Classes that must not be included in selection table
WHITE_LIST = []
BLACK_LIST = []#
####################  AAL SERVER  #####################
#MODEL_PATH=
TEST_MODELS = ['AAL_Pimodel_model_epoch_15.pkl']
MODEL_PATH = 'bb/'
SERVER_MODELS = ['AAL_Pimodel_model_epoch_20.pkl']

SERVER_MODEL_PATH = 'bb'

STREAM_MODELS = [
                 'AAL_Pimodel_model_epoch_20.pkl',
                                 
                ]

#NUC_MODELS = ['AAL_Pimodel_model_epoch_Params15.pkl']
NUC_MODELS = ['AAL_Pimodel_model_epoch_Params20.pkl']

MAX_CHUNK_SIZE = int((1024 * 16) * (SPEC_LENGTH + 2.0)) #n + 1 seconds @ 128kbps
#MAX_CHUNK_SIZE =5
RESULT_POOLING = 2 
MAX_RESULTS = 5
SCORE_MULTIPLY = 1.0

COUNTER_FILE = 'aal_count.json'

COUNT_THRESHOLD = 0.25
COUNT_TIMEOUT = 2


FRAMES_PER_BUFFER = SAMPLE_RATE // 2
PREDICTION_POOL_SIZE = 1
PREDICTION_THRESHOLD = 0.1

MEAN_TARGETS_PER_IMAGE = 1.15


MODEL_TYPE='pi'
TEST_FUNCTIONS = []
FRAMES = []
PREDICTION_STACK = []
KILL_ALL = False

####################  STATS AND LOG  ######################

# Global vars
CLASSES = []
STATS = {}
DO_BREAK = False
MAX_POOLING=True
# Options for log mode are 'all', 'info', 'progress', 'error', 'result'
LOG_MODE = 'all'



################# CONFIG EXPORT AND LOAD  #################
def getModelSettings():

    s = {}

    s['classes'] = CLASSES
    s['spec_type'] = SPEC_TYPE
    s['amplitude_scale'] = AMPLITUDE_SCALE
    s['sample_rate'] = SAMPLE_RATE
    s['spec_length'] = SPEC_LENGTH
    s['spec_fmin'] = SPEC_FMIN
    s['spec_fmax'] = SPEC_FMAX
    s['im_dim'] = IM_DIM
    s['im_size'] = IM_SIZE
    s['zero_centered_normalization'] = ZERO_CENTERED_NORMALIZATION

    return s

def setModelSettings(s):

    if 'classes' in s:
        global CLASSES
        CLASSES = s['classes']

    if 'spec_type' in s:
        global SPEC_TYPE
        SPEC_TYPE = s['spec_type'] 

    if 'amplitude_scale' in s:
        global AMPLITUDE_SCALE
        AMPLITUDE_SCALE = s['amplitude_scale']

    if 'sample_rate' in s:
        global SAMPLE_RATE
        SAMPLE_RATE = s['sample_rate']

    if 'spec_length' in s:
        global SPEC_LENGTH
        SPEC_LENGTH = s['spec_length']

    if 'spec_fmin' in s:
        global SPEC_FMIN
        SPEC_FMIN = s['spec_fmin']

    if 'spec_fmax' in s:
        global SPEC_FMAX
        SPEC_FMAX = s['spec_fmax']
        
    if 'im_dim' in s:        
        global IM_DIM
        IM_DIM = s['im_dim']

    if 'im_size' in s:
        global IM_SIZE
        IM_SIZE = s['im_size']

    if 'zero_centered_normalization' in s:
        global ZERO_CENTERED_NORMALIZATION
        ZERO_CENTERED_NORMALIZATION = s['zero_centered_normalization']
    

