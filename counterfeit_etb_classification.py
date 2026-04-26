import streamlit as st
import numpy as np
import cv2
import os
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
from PIL import Image

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ETB Banknote Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for distinctive, production-grade UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    :root {
        --primary: #10B981;
        --primary-dark: #059669;
        --secondary: #1E293B;
        --accent: #F59E0B;
        --danger: #EF4444;
        --success: #22C55E;
        --bg-dark: #0F172A;
        --bg-card: #1E293B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }
    
    * {
        font-family: 'DM Sans', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        min-height: 100vh;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Header styling */
    .header-container {
        padding: 2rem 0;
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
        margin-bottom: 2rem;
    }
    
    .title-main {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10B981, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .result-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    
    .result-card:hover {
        border-color: rgba(16, 185, 129, 0.6);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Classification result badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .badge-genuine {
        background: rgba(34, 197, 94, 0.2);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    .badge-counterfeit {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        background: #334155;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(16, 185, 129, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #10B981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    /* Warning banner */
    .warning-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Model info cards */
    .model-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
model_paths = {
    'Dense121 - New Banknotes': 'models/dense121.tflite',
    'Dense121 - Worn-out Banknotes': 'models/Dense121_best_weights_mixed-III.tflite',
    'VGG19 - New Banknotes': 'models/Vgg19.tflite',
    'VGG19 - Worn-out Banknotes': 'models/Vgg19_mixed.tflite',
}

classes = ['genuine_200_etb', 'counterfeit_200_etb', 'genuine_100_etb', 'counterfeit_100_etb']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_resource
def load_selected_model(model_path):
    """Load TensorFlow Lite model with caching"""
    try:
        # Try tflite_runtime first (lighter weight)
        interpreter = tflite.Interpreter(model_path=model_path)
    except (AttributeError, NameError):
        # Fallback to tensorflow.lite
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    return image

def predict_image(model, image):
    """Make predictions on the image"""
    processed_image = preprocess_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], expanded_image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])
    
    top_classes = np.argsort(predictions, axis=1)[0][-4:][::-1]
    top_confidences = predictions[0][top_classes]
    
    return top_classes, top_confidences

def format_class_name(class_name):
    """Format class name for display"""
    return class_name.replace('_', ' ').title()

def get_result_badge(class_name, confidence):
    """Generate HTML badge for classification result"""
    is_genuine = 'genuine' in class_name.lower()
    badge_class = 'badge-genuine' if is_genuine else 'badge-counterfeit'
    icon = '✓' if is_genuine else '✗'
    
    return f"""
    <div class="result-card animate-in">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span class="badge {badge_class}">{icon} {format_class_name(class_name)}</span>
            <span style="color: #94A3B8; font-size: 0.9rem;">{confidence * 100:.1f}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence * 100}%"></div>
        </div>
    </div>
    """

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="title-main">🛡️ ETB Banknote Classifier</h1>
        <p class="subtitle">Master's Thesis Research • Counterfeit Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Warning banner
    st.markdown("""
    <div class="warning-banner">
        <p style="margin: 0; color: #F59E0B; font-weight: 500;">
            ⚠️ <strong>Experimental Model</strong> — Not ready for real-world application. Use at your own risk.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        selected_model = st.selectbox(
            "Select Model & Condition",
            list(model_paths.keys()),
            help="Choose the appropriate model based on banknote condition"
        )
        
        model_info = {
            'Dense121 - New Banknotes': 'Best for crisp, new currency notes',
            'Dense121 - Worn-out Banknotes': 'Optimized for aged/worn banknotes',
            'VGG19 - New Banknotes': 'VGG architecture for new notes',
            'VGG19 - Worn-out Banknotes': 'VGG architecture for worn notes'
        }
        
        st.info(model_info[selected_model])
        
        st.markdown("---")
        st.markdown("### 💡 Supported Classes")
        for cls in classes:
            icon = '✓' if 'genuine' in cls else '✗'
            st.markdown(f"- {icon} {format_class_name(cls)}")
    
    # Input source selection
    input_source = st.radio("📥 Select Input Source", ["Upload Images", "Use Sample Data"], horizontal=True)
    
    uploaded_files = []
    sample_files = []
    
    if input_source == "Upload Images":
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload images of Ethiopian banknotes (100 or 200 ETB)"
        )
    else:
        # Sample data selection
        sample_dir = "sample_data"
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Group by category
        sample_options = ["Select all"] + sorted(sample_images)
        selected_samples = st.multiselect("Choose sample images", sample_options, default=["Select all"])
        
        if "Select all" in selected_samples:
            sample_files = [os.path.join(sample_dir, f) for f in sample_images]
        else:
            sample_files = [os.path.join(sample_dir, f) for f in selected_samples]
        
        if sample_files:
            st.markdown(f"**{len(sample_files)} sample(s) selected**")
    
    # Classification button
    has_files = uploaded_files or sample_files
    if has_files and st.button("🚀 Classify", type="primary", use_container_width=True):
        # Load model
        model_path = model_paths[selected_model]
        
        with st.spinner("Loading model..."):
            model = load_selected_model(model_path)
        
        # Process files
        files_to_process = uploaded_files if uploaded_files else sample_files
        
        st.markdown("---")
        
        # Process each image
        for idx, file_item in enumerate(files_to_process):
            # Read image
            if uploaded_files:
                image = cv2.imdecode(np.frombuffer(file_item.read(), np.uint8), 1)
                file_name = file_item.name
            else:
                image = cv2.imread(file_item)
                file_name = os.path.basename(file_item)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            top_classes, top_confidences = predict_image(model, image)
            
            # Get top prediction
            top_class = classes[top_classes[0]]
            top_confidence = top_confidences[0]
            is_genuine = 'genuine' in top_class.lower()
            
            # Compact result display
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col1:
                resized_image = cv2.resize(image_rgb, (150, 100))
                st.image(resized_image, use_container_width=True)
            
            with result_col2:
                st.markdown(f"**{file_name}**")
                # Show top 2 predictions
                for i in range(min(2, len(top_classes))):
                    cls_name = classes[top_classes[i]]
                    conf = top_confidences[i]
                    icon = '✓' if 'genuine' in cls_name.lower() else '✗'
                    color = '#22C55E' if 'genuine' in cls_name.lower() else '#EF4444'
                    st.markdown(f"<span style='color:{color}'>{icon}</span> {format_class_name(cls_name)}: **{conf*100:.1f}%**", unsafe_allow_html=True)
            
            with result_col3:
                # Status badge
                if is_genuine:
                    st.success(f"✓ GENUINE\n{top_confidence*100:.1f}%")
                else:
                    st.error(f"✗ COUNTERFEIT\n{top_confidence*100:.1f}%")
            
            st.markdown("---")
        
        # Summary
        st.success(f"✅ Classification complete for {len(files_to_process)} image(s)")

if __name__ == '__main__':
    main()
