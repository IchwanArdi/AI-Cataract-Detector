import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATHS = [
    'best_cataract_model_fixed.h5',
    'cataract_model_new.h5', 
    'best_cataract_model.h5'
]

# Default model parameters
DEFAULT_TARGET_SIZE = (224, 224)
DEFAULT_CHANNELS = 3
CLASS_INDICES = {'cataract': 0, 'normal': 1}

# UI Configuration
PAGE_CONFIG = {
    "page_title": "Cataract Detection System",
    "page_icon": "👁️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Default settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png']

# Version info
APP_VERSION = "2.0"
APP_NAME = "Cataract Detection System"