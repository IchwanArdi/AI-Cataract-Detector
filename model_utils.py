import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array
import numpy as np
from PIL import Image
from config import MODEL_PATHS, DEFAULT_TARGET_SIZE, DEFAULT_CHANNELS


@st.cache_resource
def load_model():
    """Load model with enhanced error handling"""
    
    for model_path in MODEL_PATHS:
        try:
            st.info(f"🔄 Mencoba memuat model dari: {model_path}")
            
            # Try loading with different methods
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception as e1:
                st.warning(f"Method 1 failed: {str(e1)}")
                # Try with compile=False
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    st.info("✅ Model dimuat tanpa kompilasi")
                except Exception as e2:
                    st.warning(f"Method 2 failed: {str(e2)}")
                    continue
            
            # Get model input shape
            input_shape = model.input_shape
            st.success(f"✅ Model berhasil dimuat dari: {model_path}")
            st.info(f"📏 Model input shape: {input_shape}")
            
            # Validate input shape
            if len(input_shape) == 4:
                channels = input_shape[3]
                height = input_shape[1] 
                width = input_shape[2]
                st.info(f"📊 Detected channels: {channels}, Size: {height}x{width}")
                
                return model, True, input_shape
            else:
                st.warning(f"⚠️ Unexpected input shape: {input_shape}")
                return model, True, input_shape
                
        except Exception as e:
            st.warning(f"❌ Failed to load {model_path}: {str(e)}")
            continue
    
    # If all model paths fail, show detailed error
    st.error("❌ Tidak dapat memuat model dari path manapun")
    st.error("📝 Daftar path yang dicoba:")
    for path in MODEL_PATHS:
        st.write(f"  - {path}")
    
    st.markdown("""
    ### 🔧 Possible Solutions:
    1. **Check model file location** - Pastikan file model ada di direktori yang benar
    2. **Re-train model** - Model mungkin corrupt atau tidak kompatibel
    3. **Check TensorFlow version** - Pastikan versi TF sama antara training dan deployment
    4. **Try different model format** - Coba SavedModel format instead of .h5
    """)
    
    return None, False, None


def preprocess_image(image, target_size=DEFAULT_TARGET_SIZE, target_channels=DEFAULT_CHANNELS):
    """Enhanced preprocess image for model prediction with channel handling"""
    
    # Convert image based on target channels
    if target_channels == 1:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
    elif target_channels == 3:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        st.warning(f"⚠️ Unexpected channel count: {target_channels}")
        # Default to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    # Resize image to match model input
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    st.info(f"📊 Preprocessed image shape: {img_array.shape}")
    
    return img_array


def predict_cataract(model, image, class_indices, input_shape):
    """Enhanced prediction function with better error handling"""
    
    try:
        # Extract target size and channels from input shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            target_size = (input_shape[1], input_shape[2])
            target_channels = input_shape[3]
        else:
            target_size = DEFAULT_TARGET_SIZE
            target_channels = DEFAULT_CHANNELS
        
        st.info(f"🎯 Target size: {target_size}, Channels: {target_channels}")
        
        img_array = preprocess_image(image, target_size, target_channels)
        
        # Validate preprocessed image shape
        expected_shape = (1, target_size[0], target_size[1], target_channels)
        if img_array.shape != expected_shape:
            st.error(f"❌ Shape mismatch: Expected {expected_shape}, Got {img_array.shape}")
            return None, None, None
        
        # Get prediction
        with st.spinner('🔮 Making prediction...'):
            pred = model.predict(img_array, verbose=0)
            
        st.info(f"📈 Raw prediction output: {pred}")
        
        # Handle different prediction output formats
        if isinstance(pred, (list, tuple)):
            pred_value = pred[0][0] if len(pred[0]) == 1 else pred[0]
        elif pred.ndim == 2 and pred.shape[1] == 1:
            pred_value = pred[0][0]
        elif pred.ndim == 2 and pred.shape[1] == 2:
            # Binary classification with 2 outputs
            pred_value = pred[0]
        else:
            pred_value = pred[0] if pred.ndim > 1 else pred
        
        # Calculate probabilities based on output format
        if isinstance(pred_value, (list, np.ndarray)) and len(pred_value) == 2:
            # Two-class output
            prob_class0 = float(pred_value[0])
            prob_class1 = float(pred_value[1])
            
            # Normalize if needed
            total = prob_class0 + prob_class1
            if total > 0:
                prob_class0 /= total
                prob_class1 /= total
        else:
            # Single output (sigmoid)
            pred_value = float(pred_value)
            prob_class0 = pred_value
            prob_class1 = 1.0 - pred_value
        
        # Map to cataract/normal based on class indices
        if class_indices.get('cataract', 0) == 0:
            prob_cataract = prob_class0
            prob_normal = prob_class1
        else:
            prob_cataract = prob_class1
            prob_normal = prob_class0
        
        return pred_value, prob_cataract, prob_normal
        
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.error(f"🔍 Debug info: Input shape: {input_shape}, Image mode: {image.mode}")
        return None, None, None