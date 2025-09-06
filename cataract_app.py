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

# Load model function with caching
@st.cache_resource
def load_model():
    try:
        # Try to load the best model first
        model = tf.keras.models.load_model('best_cataract_model_v2.h5')
        st.success("✅ Model berhasil dimuat!")
        
        # Get model input shape
        input_shape = model.input_shape
        st.info(f"📏 Model input shape: {input_shape}")
        
        return model, True, input_shape
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.warning("⚠️ Pastikan path model sudah benar dan model sudah di-training.")
        return None, False, None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function
def predict_cataract(model, image, class_indices, input_shape):
    """Make prediction on preprocessed image"""
    # Extract target size from input shape
    if len(input_shape) == 4:  # (batch, height, width, channels)
        target_size = (input_shape[1], input_shape[2])
    else:
        target_size = (224, 224)  # fallback
    
    img_array = preprocess_image(image, target_size)
    
    # Get prediction
    pred = model.predict(img_array, verbose=0)[0][0]
    
    # Calculate probabilities based on class mapping
    if class_indices.get('cataract', 0) == 0:
        prob_cataract = pred  # Changed: direct assignment
        prob_normal = 1.0 - pred
    else:
        prob_cataract = 1.0 - pred
        prob_normal = pred
    
    return pred, prob_cataract, prob_normal

# Visualization functions
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

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Cataract Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar mata untuk deteksi katarak menggunakan AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading AI model...'):
        model, model_loaded, input_shape = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar for information and settings
    with st.sidebar:
        st.header("ℹ️ Informasi Sistem")
        st.markdown(f"""
        **Cataract Detection AI v1.0**
        
        **Model Input Size:** {input_shape[1]}x{input_shape[2]} pixels
        
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
            """)
    
    with col2:
        st.header("🔍 Hasil Analisis")
        
        if uploaded_file is not None:
            with st.spinner('Menganalisis gambar...'):
                try:
                    # Make prediction
                    class_indices = {'cataract': 0, 'normal': 1}  # Default mapping
                    raw_pred, prob_cataract, prob_normal = predict_cataract(model, image, class_indices, input_shape)
                    
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
                    st.error(f"Error dalam prediksi: {str(e)}")
                    st.info("💡 Tip: Pastikan model sudah di-training dengan benar")
                    st.stop()
        else:
            st.info("👆 Upload gambar untuk memulai analisis")
    
    # Charts section
    if uploaded_file is not None and 'prob_normal' in locals():
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
                **Raw Prediction Value:** {raw_pred:.6f}
                **Processing Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                **Probability Breakdown:**
                - Normal: {prob_normal:.4f} ({prob_normal*100:.2f}%)
                - Cataract: {prob_cataract:.4f} ({prob_cataract*100:.2f}%)
                
                **Class Mapping:**
                - Class 0: Cataract
                - Class 1: Normal
                """)
            
            with st.expander("Image Processing Details", expanded=False):
                target_size = (input_shape[1], input_shape[2]) if input_shape else (224, 224)
                st.markdown(f"""
                **Original Size:** {image.size[0]} x {image.size[1]}
                **Processed Size:** {target_size[0]} x {target_size[1]}
                **Normalization:** Pixel values scaled to [0, 1]
                **Color Mode:** RGB
                **Preprocessing:** Resize + Normalize
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; margin-top: 2rem;">
        <p><strong>⚠️ Disclaimer:</strong> Sistem ini adalah alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional.</p>
        <p>Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat.</p>
        <p>Cataract Detection System © 2025 | Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Download results function
def download_results(prediction_label, confidence, prob_normal, prob_cataract, recommendation):
    """Generate downloadable report"""
    
    report = f"""
    CATARACT DETECTION REPORT
    ========================
    
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PREDICTION RESULTS:
    - Final Prediction: {prediction_label}
    - Confidence Level: {confidence*100:.1f}%
    
    PROBABILITY BREAKDOWN:
    - Normal Eye: {prob_normal*100:.2f}%
    - Cataract: {prob_cataract*100:.2f}%
    
    RECOMMENDATION:
    {recommendation}
    
    IMPORTANT NOTICE:
    This system is a diagnostic aid and does not replace professional medical consultation.
    Always consult with an eye doctor for accurate diagnosis.
    
    ========================
    Cataract Detection System v1.0
    """
    
    return report

if __name__ == "__main__":
    main()