"""
Main application file for Cataract Detection System
Modular version with separated components
"""

import streamlit as st

# Import local modules
from config import PAGE_CONFIG, CLASS_INDICES
from styles import CUSTOM_CSS
from model_utils import load_model, predict_cataract
from visualization import create_probability_chart, create_confidence_gauge
from ui_components import (
    render_sidebar, 
    render_file_uploader, 
    render_prediction_result,
    render_technical_details,
    render_debug_info,
    render_footer
)
from utils import (
    determine_prediction_result, 
    extract_model_parameters,
    validate_image_info
)


def setup_page():
    """Setup page configuration and styles"""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">Cataract Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar mata untuk deteksi katarak menggunakan AI</p>', unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Setup
    setup_page()
    render_header()
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, model_loaded, input_shape = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Render sidebar and get settings
    confidence_threshold, show_technical_details, debug_mode = render_sidebar(input_shape)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload section
        uploaded_file, image, image_info = render_file_uploader()
        
        # Debug info if enabled
        if uploaded_file is not None and debug_mode:
            render_debug_info(image_info, input_shape)
    
    with col2:
        # Results section
        st.header("🔍 Hasil Analisis")
        
        if uploaded_file is not None:
            # Validate image
            is_valid, validation_message = validate_image_info(image_info)
            
            if not is_valid:
                st.error(f"❌ {validation_message}")
                st.stop()
            
            # Make prediction
            with st.spinner('🔍 Menganalisis gambar...'):
                try:
                    raw_pred, prob_cataract, prob_normal = predict_cataract(
                        model, image, CLASS_INDICES, input_shape
                    )
                    
                    if raw_pred is None:
                        st.error("❌ Gagal melakukan prediksi")
                        st.stop()
                    
                    # Determine prediction result
                    prediction_label, recommendation, max_prob = determine_prediction_result(
                        prob_cataract, prob_normal, confidence_threshold
                    )
                    
                    # Render prediction result
                    render_prediction_result(
                        prediction_label, max_prob, recommendation, confidence_threshold
                    )
                    
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
            target_size, target_channels = extract_model_parameters(input_shape)
            
            render_technical_details(
                raw_pred, prob_normal, prob_cataract, input_shape, 
                image, target_size, target_channels
            )
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()