from config import DEFAULT_TARGET_SIZE, DEFAULT_CHANNELS


def determine_prediction_result(prob_cataract, prob_normal, confidence_threshold):
    """Determine prediction label and recommendation based on probabilities"""
    
    max_prob = max(prob_cataract, prob_normal)
    
    if max_prob < confidence_threshold:
        prediction_label = "Uncertain"
        recommendation = "Konsultasi dengan dokter mata direkomendasikan"
    elif prob_cataract > prob_normal:
        prediction_label = "Cataract Detected"
        recommendation = "Segera konsultasi dengan dokter mata"
    else:
        prediction_label = "Normal"
        recommendation = "Mata terlihat normal, tetap jaga kesehatan mata"
    
    return prediction_label, recommendation, max_prob


def extract_model_parameters(input_shape):
    """Extract target size and channels from model input shape"""
    
    if input_shape and len(input_shape) == 4:
        target_size = (input_shape[1], input_shape[2])
        target_channels = input_shape[3]
    else:
        target_size = DEFAULT_TARGET_SIZE
        target_channels = DEFAULT_CHANNELS
    
    return target_size, target_channels


def validate_image_info(image_info):
    """Validate uploaded image information"""
    
    if not image_info:
        return False, "No image uploaded"
    
    min_size = 100  # minimum acceptable size
    width, height = image_info['size']
    
    if width < min_size or height < min_size:
        return False, f"Image too small. Minimum size: {min_size}x{min_size}"
    
    return True, "Image valid"