import streamlit as st
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cataract Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .normal-prediction {
        background-color: #D5F4E6;
        border: 2px solid #27AE60;
        color: #1E8449;
    }
    .cataract-prediction {
        background-color: #FADBD8;
        border: 2px solid #E74C3C;
        color: #C0392B;
    }
    .uncertain-prediction {
        background-color: #FEF9E7;
        border: 2px solid #F39C12;
        color: #D68910;
    }
    .info-box {
        background-color: #008000;
        border-left: 5px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FDF2E9;
        border-left: 5px solid #E67E22;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function with enhanced error handling
@st.cache_resource
def load_model():
    model_paths = [ 'best_cataract_model_fixed.h5', 'cataract_model_new.h5', 'best_cataract_model.h5' ]
    
    for model_path in model_paths:
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
    for path in model_paths:
        st.write(f"  - {path}")
    
    st.markdown("""
    ### 🔧 Possible Solutions:
    1. **Check model file location** - Pastikan file model ada di direktori yang benar
    2. **Re-train model** - Model mungkin corrupt atau tidak kompatibel
    3. **Check TensorFlow version** - Pastikan versi TF sama antara training dan deployment
    4. **Try different model format** - Coba SavedModel format instead of .h5
    """)
    
    return None, False, None

# Enhanced image preprocessing function
def preprocess_image(image, target_size=(224, 224), target_channels=3):
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

# Enhanced prediction function
def predict_cataract(model, image, class_indices, input_shape):
    """Enhanced prediction function with better error handling"""
    
    try:
        # Extract target size and channels from input shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            target_size = (input_shape[1], input_shape[2])
            target_channels = input_shape[3]
        else:
            target_size = (224, 224)  # fallback
            target_channels = 3  # fallback to RGB
        
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

# Visualization functions (unchanged)
def create_probability_chart(prob_normal, prob_cataract):
    """Create interactive probability chart"""
    
    fig = go.Figure(data=[
        go.Bar(
            name='Probability',
            x=['Normal', 'Cataract'],
            y=[prob_normal, prob_cataract],
            marker_color=['#27AE60', '#E74C3C'],
            text=[f'{prob_normal*100:.1f}%', f'{prob_cataract*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        yaxis_title='Probability',
        xaxis_title='Class',
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Enhanced main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Cataract Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar mata untuk deteksi katarak menggunakan AI</p>', unsafe_allow_html=True)
    
    # Load model with enhanced error handling
    with st.spinner('🔄 Loading AI model...'):
        model, model_loaded, input_shape = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar for information and settings
    with st.sidebar:
        st.header("ℹ️ Informasi Sistem")
        if input_shape:
            channels = input_shape[3] if len(input_shape) == 4 else "Unknown"
            st.markdown(f"""
            **Cataract Detection AI v2.0**
            
            **Model Info:**
            - Input Size: {input_shape[1] if len(input_shape) > 1 else 'Unknown'}x{input_shape[2] if len(input_shape) > 2 else 'Unknown'} pixels
            - Channels: {channels}
            - Format: {'Grayscale' if channels == 1 else 'RGB' if channels == 3 else 'Unknown'}
            
            **Cara Penggunaan:**
            1. Upload gambar mata
            2. Tunggu proses analisis
            3. Lihat hasil prediksi
            
            **Format Gambar:**
            - JPG, JPEG, PNG
            - Resolusi minimal 224x224px
            - Gambar mata yang jelas
            """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Minimum confidence untuk prediksi yang dianggap valid"
        )
        
        show_technical_details = st.checkbox(
            "Show Technical Details",
            value=False,
            help="Tampilkan detail teknis prediksi"
        )
        
        # Debug option
        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Tampilkan informasi debug tambahan"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Gambar")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar mata untuk dianalisis:",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar mata dengan format JPG, JPEG, atau PNG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            **Informasi Gambar:**
            - Nama file: {uploaded_file.name}
            - Ukuran: {image.size[0]} x {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            - Channels: {len(image.getbands()) if hasattr(image, 'getbands') else 'Unknown'}
            """)
            
            if debug_mode:
                st.markdown("**Debug Info:**")
                st.write(f"Image bands: {image.getbands() if hasattr(image, 'getbands') else 'N/A'}")
                st.write(f"Model expects: {input_shape}")
    
    with col2:
        st.header("🔍 Hasil Analisis")
        
        if uploaded_file is not None:
            with st.spinner('🔍 Menganalisis gambar...'):
                try:
                    # Make prediction
                    class_indices = {'cataract': 0, 'normal': 1}  # Default mapping
                    raw_pred, prob_cataract, prob_normal = predict_cataract(model, image, class_indices, input_shape)
                    
                    if raw_pred is None:
                        st.error("❌ Gagal melakukan prediksi")
                        st.stop()
                    
                    # Determine final prediction
                    max_prob = max(prob_cataract, prob_normal)
                    
                    if max_prob < confidence_threshold:
                        prediction_label = "Uncertain"
                        prediction_class = "uncertain-prediction"
                        recommendation = "Konsultasi dengan dokter mata direkomendasikan"
                        emoji = "⚠️"
                    elif prob_cataract > prob_normal:
                        prediction_label = "Cataract Detected"
                        prediction_class = "cataract-prediction"
                        recommendation = "Segera konsultasi dengan dokter mata"
                        emoji = "🔴"
                    else:
                        prediction_label = "Normal"
                        prediction_class = "normal-prediction"
                        recommendation = "Mata terlihat normal, tetap jaga kesehatan mata"
                        emoji = "✅"
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        {emoji} {prediction_label}<br>
                        Confidence: {max_prob*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display recommendation
                    if prediction_label == "Cataract Detected":
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>⚠️ Rekomendasi:</strong><br>
                            {recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>ℹ️ Rekomendasi:</strong><br>
                            {recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {str(e)}")
                    if debug_mode:
                        st.exception(e)
                    st.info("💡 Tips untuk mengatasi error:")
                    st.write("- Pastikan gambar dalam format yang benar")
                    st.write("- Coba gambar dengan ukuran yang berbeda")
                    st.write("- Re-train model jika masalah berlanjut")
                    st.stop()
        else:
            st.info("👆 Upload gambar untuk memulai analisis")
    
    # Charts section
    if uploaded_file is not None and 'prob_normal' in locals() and prob_normal is not None:
        st.header("📊 Visualisasi Hasil")
        
        chart_col1, chart_col2 = st.columns([1, 1])
        
        with chart_col1:
            # Probability chart
            prob_fig = create_probability_chart(prob_normal, prob_cataract)
            st.plotly_chart(prob_fig, use_container_width=True)
        
        with chart_col2:
            # Confidence gauge
            confidence_fig = create_confidence_gauge(max_prob)
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Technical details
        if show_technical_details:
            st.header("🔧 Technical Details")
            
            with st.expander("Model Information", expanded=False):
                st.markdown(f"""
                **Model Input Shape:** {input_shape}
                **Raw Prediction Value:** {raw_pred}
                **Processing Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                **Probability Breakdown:**
                - Normal: {prob_normal:.4f} ({prob_normal*100:.2f}%)
                - Cataract: {prob_cataract:.4f} ({prob_cataract*100:.2f}%)
                
                **Class Mapping:**
                - Class 0: Cataract
                - Class 1: Normal
                """)
            
            with st.expander("Image Processing Details", expanded=False):
                if input_shape and len(input_shape) == 4:
                    target_size = (input_shape[1], input_shape[2])
                    target_channels = input_shape[3]
                else:
                    target_size = (224, 224)
                    target_channels = 3
                
                st.markdown(f"""
                **Original Size:** {image.size[0]} x {image.size[1]}
                **Processed Size:** {target_size[0]} x {target_size[1]}
                **Original Channels:** {len(image.getbands()) if hasattr(image, 'getbands') else 'Unknown'}
                **Target Channels:** {target_channels}
                **Normalization:** Pixel values scaled to [0, 1]
                **Color Mode:** {'Grayscale' if target_channels == 1 else 'RGB'}
                **Preprocessing:** Resize + Color Convert + Normalize
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; margin-top: 2rem;">
        <p><strong>⚠️ Disclaimer:</strong> Sistem ini adalah alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional.</p>
        <p>Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat.</p>
        <p>Cataract Detection System v2.0 © 2025 | Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()