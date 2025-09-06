import streamlit as st
from datetime import datetime
from PIL import Image
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD, 
    SUPPORTED_IMAGE_TYPES, 
    APP_VERSION, 
    APP_NAME
)


def render_sidebar(input_shape):
    """Render sidebar with information and settings"""
    
    with st.sidebar:
        st.header("ℹ️ Informasi Sistem")
        if input_shape:
            channels = input_shape[3] if len(input_shape) == 4 else "Unknown"
            st.markdown(f"""
            **{APP_NAME} v{APP_VERSION}**
            
            **Model Info:**
            - Input Size: {input_shape[1] if len(input_shape) > 1 else 'Unknown'}x{input_shape[2] if len(input_shape) > 2 else 'Unknown'} pixels
            - Channels: {channels}
            - Format: {'Grayscale' if channels == 1 else 'RGB' if channels == 3 else 'Unknown'}
            
            **Cara Penggunaan:**
            1. Upload gambar mata
            2. Tunggu proses analisis
            3. Lihat hasil prediksi
            
            **Format Gambar:**
            - {', '.join(SUPPORTED_IMAGE_TYPES).upper()}
            - Resolusi minimal 224x224px
            - Gambar mata yang jelas
            """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum confidence untuk prediksi yang dianggap valid"
        )
        
        show_technical_details = st.checkbox(
            "Show Technical Details",
            value=False,
            help="Tampilkan detail teknis prediksi"
        )
        
        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Tampilkan informasi debug tambahan"
        )
    
    return confidence_threshold, show_technical_details, debug_mode


def render_file_uploader():
    """Render file uploader section and return uploaded file and image info"""
    
    st.header("📁 Upload Gambar")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar mata untuk dianalisis:",
        type=SUPPORTED_IMAGE_TYPES,
        help=f"Upload gambar mata dengan format {', '.join(SUPPORTED_IMAGE_TYPES).upper()}"
    )
    
    image_info = None
    image = None
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        
        # Image info
        image_info = {
            'filename': uploaded_file.name,
            'size': image.size,
            'format': image.format,
            'mode': image.mode,
            'channels': len(image.getbands()) if hasattr(image, 'getbands') else 'Unknown'
        }
        
        st.markdown(f"""
        **Informasi Gambar:**
        - Nama file: {image_info['filename']}
        - Ukuran: {image_info['size'][0]} x {image_info['size'][1]} pixels
        - Format: {image_info['format']}
        - Mode: {image_info['mode']}
        - Channels: {image_info['channels']}
        """)
    
    return uploaded_file, image, image_info


def render_prediction_result(prediction_label, max_prob, recommendation, confidence_threshold):
    """Render prediction result box"""
    
    if max_prob < confidence_threshold:
        prediction_class = "uncertain-prediction"
        emoji = "⚠️"
    elif prediction_label == "Cataract Detected":
        prediction_class = "cataract-prediction"
        emoji = "🔴"
    else:
        prediction_class = "normal-prediction"
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


def render_technical_details(raw_pred, prob_normal, prob_cataract, input_shape, image, target_size, target_channels):
    """Render technical details section"""
    
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
        st.markdown(f"""
        **Original Size:** {image.size[0]} x {image.size[1]}
        **Processed Size:** {target_size[0]} x {target_size[1]}
        **Original Channels:** {len(image.getbands()) if hasattr(image, 'getbands') else 'Unknown'}
        **Target Channels:** {target_channels}
        **Normalization:** Pixel values scaled to [0, 1]
        **Color Mode:** {'Grayscale' if target_channels == 1 else 'RGB'}
        **Preprocessing:** Resize + Color Convert + Normalize
        """)


def render_debug_info(image_info, input_shape):
    """Render debug information"""
    
    st.markdown("**Debug Info:**")
    if image_info:
        st.write(f"Image bands: {image_info.get('channels', 'N/A')}")
    st.write(f"Model expects: {input_shape}")


def render_footer():
    """Render application footer"""
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7F8C8D; margin-top: 2rem;">
        <p><strong>⚠️ Disclaimer:</strong> Sistem ini adalah alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional.</p>
        <p>Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat.</p>
        <p>{APP_NAME} v{APP_VERSION} © 2025 | Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)