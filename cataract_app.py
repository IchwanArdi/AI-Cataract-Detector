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
        background-color: #EBF5FB;
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
    .error-box {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced model loading function with multiple fallback strategies
@st.cache_resource
def load_model():
    """Load model with enhanced error handling and fallback strategies"""
    model_paths = [
        'best_cataract_model_v2.h5',
        'cataract_model.h5',
        'model.h5',
        'cataract_detection_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Method 1: Try loading with compile=False
                st.info(f"🔄 Attempting to load model: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Get model details
                input_shape = model.input_shape
                total_params = model.count_params()
                
                st.success(f"✅ Model berhasil dimuat dari: {model_path}")
                st.info(f"📏 Model input shape: {input_shape}")
                st.info(f"🔢 Total parameters: {total_params:,}")
                
                # Verify model compatibility
                if len(input_shape) >= 4:
                    channels = input_shape[-1] if input_shape[-1] is not None else 3
                    st.info(f"🎨 Input channels detected: {channels}")
                    return model, True, input_shape, channels
                else:
                    st.warning("⚠️ Model architecture tidak standar, menggunakan default channels=3")
                    return model, True, input_shape, 3
                    
            except Exception as e:
                st.warning(f"⚠️ Gagal memuat {model_path}: {str(e)}")
                continue
    
    # If no model found, show comprehensive error message
    st.error("❌ Tidak dapat memuat model apapun!")
    st.markdown("""
    <div class="error-box">
        <strong>🚨 Model Loading Failed</strong><br><br>
        <strong>Kemungkinan penyebab:</strong><br>
        1. File model tidak ditemukan<br>
        2. Model dilatih dengan versi TensorFlow yang berbeda<br>
        3. Architecture mismatch (input channels)<br>
        4. Model corrupt atau rusak<br><br>
        <strong>Solusi yang dapat dicoba:</strong><br>
        1. Pastikan file model ada di direktori yang sama<br>
        2. Re-train model dengan konsistensi input shape<br>
        3. Konversi model ke format yang kompatibel<br>
        4. Periksa versi TensorFlow yang digunakan
    </div>
    """, unsafe_allow_html=True)
    
    return None, False, None, None

# Enhanced image preprocessing with flexible channel handling
def preprocess_image(image, target_size=(224, 224), target_channels=3):
    """Enhanced preprocessing with flexible channel handling"""
    try:
        # Handle different image modes
        if target_channels == 1:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
        elif target_channels == 3:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = img_to_array(image)
        
        # Handle channel dimension for grayscale
        if target_channels == 1 and len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array[:,:,0], axis=-1)
        elif target_channels == 3 and len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, True
        
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None, False

# Enhanced prediction function
def predict_cataract(model, image, input_shape, target_channels):
    """Make prediction with enhanced error handling"""
    try:
        # Extract target size from input shape
        if len(input_shape) >= 4:
            target_size = (input_shape[1] or 224, input_shape[2] or 224)
        else:
            target_size = (224, 224)
        
        # Preprocess image
        img_array, preprocessing_success = preprocess_image(image, target_size, target_channels)
        
        if not preprocessing_success:
            return None, None, None, False
        
        # Make prediction
        with st.spinner('🔮 Making prediction...'):
            pred = model.predict(img_array, verbose=0)
            
            # Handle different output shapes
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                # Multi-class output
                pred_probs = pred[0]
                if len(pred_probs) == 2:
                    prob_normal = pred_probs[0]
                    prob_cataract = pred_probs[1]
                else:
                    # Binary classification with single output
                    prob_cataract = pred_probs[0]
                    prob_normal = 1.0 - prob_cataract
            else:
                # Single output (binary classification)
                raw_pred = float(pred[0][0])
                prob_cataract = raw_pred
                prob_normal = 1.0 - raw_pred
            
            return raw_pred if 'raw_pred' in locals() else prob_cataract, prob_cataract, prob_normal, True
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None, False

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

# Debug information display
def show_debug_info(model, input_shape, target_channels):
    """Display debug information for troubleshooting"""
    with st.expander("🔧 Debug Information", expanded=False):
        st.markdown("### Model Architecture Info")
        try:
            st.text(f"Input Shape: {input_shape}")
            st.text(f"Target Channels: {target_channels}")
            st.text(f"Model Type: {type(model).__name__}")
            
            if hasattr(model, 'layers') and len(model.layers) > 0:
                st.text(f"First Layer: {model.layers[0].name}")
                if hasattr(model.layers[0], 'input_shape'):
                    st.text(f"First Layer Input: {model.layers[0].input_shape}")
            
            # Try to get model summary
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            summary_text = '\n'.join(summary_list[:20])  # First 20 lines only
            st.text("Model Summary (truncated):")
            st.code(summary_text)
            
        except Exception as e:
            st.text(f"Debug info error: {str(e)}")

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Cataract Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar mata untuk deteksi katarak menggunakan AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, model_loaded, input_shape, target_channels = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar for information and settings
    with st.sidebar:
        st.header("ℹ️ Informasi Sistem")
        st.markdown(f"""
        **Cataract Detection AI v2.0**
        
        **Model Details:**
        - Input Size: {input_shape[1] if input_shape and len(input_shape) > 1 else 'Unknown'}x{input_shape[2] if input_shape and len(input_shape) > 2 else 'Unknown'} pixels
        - Channels: {target_channels}
        - Type: {'Grayscale' if target_channels == 1 else 'RGB'}
        
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
        
        show_debug = st.checkbox(
            "Show Debug Info",
            value=False,
            help="Tampilkan informasi debug model"
        )
    
    # Show debug info if requested
    if show_debug:
        show_debug_info(model, input_shape, target_channels)
    
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
            - Target Mode: {'Grayscale' if target_channels == 1 else 'RGB'}
            """)
    
    with col2:
        st.header("🔍 Hasil Analisis")
        
        if uploaded_file is not None:
            with st.spinner('🔮 Menganalisis gambar...'):
                # Make prediction
                raw_pred, prob_cataract, prob_normal, prediction_success = predict_cataract(
                    model, image, input_shape, target_channels
                )
                
                if prediction_success:
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
                    
                    # Charts section
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
                        
                        with st.expander("Prediction Details", expanded=False):
                            st.markdown(f"""
                            **Model Configuration:**
                            - Input Shape: {input_shape}
                            - Target Channels: {target_channels}
                            - Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            **Raw Prediction:** {raw_pred:.6f}
                            
                            **Probability Breakdown:**
                            - Normal: {prob_normal:.4f} ({prob_normal*100:.2f}%)
                            - Cataract: {prob_cataract:.4f} ({prob_cataract*100:.2f}%)
                            
                            **Classification Logic:**
                            - Confidence Threshold: {confidence_threshold}
                            - Max Probability: {max_prob:.4f}
                            - Decision: {prediction_label}
                            """)
                        
                        with st.expander("Image Processing Details", expanded=False):
                            target_size = (input_shape[1] or 224, input_shape[2] or 224) if input_shape else (224, 224)
                            st.markdown(f"""
                            **Image Transformation:**
                            - Original Size: {image.size[0]} x {image.size[1]}
                            - Processed Size: {target_size[0]} x {target_size[1]}
                            - Color Conversion: {image.mode} → {'L' if target_channels == 1 else 'RGB'}
                            - Normalization: [0, 255] → [0, 1]
                            - Resampling: LANCZOS
                            """)
                
                else:
                    st.error("❌ Gagal melakukan prediksi. Silakan coba lagi dengan gambar yang berbeda.")
        else:
            st.info("👆 Upload gambar untuk memulai analisis")
    
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