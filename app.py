# STREAMLIT AI VS HUMAN DETECTION APP
# ===================================

import os
import re
import time
from io import BytesIO
import nltk
import docx
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
from text_utils import preprocess

# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .ai-result {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .human-result {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

def load_models():
    models = {}
    try:
        # Load only the classification models
        model_files = {
            'svm': 'models/svm_model.pkl',
            'decision_tree': 'models/decision_tree_model.pkl',
            'adaboost': 'models/adaboost_model.pkl'
        }
        
        for name, path in model_files.items():
            try:
                models[name] = joblib.load(path)
                models[f"{name}_available"] = True
            except:
                models[f"{name}_available"] = False
        
        available_count = sum(1 for name in ['svm', 'decision_tree', 'adaboost'] 
                         if models.get(f"{name}_available", False))
        
        models['any_model_available'] = available_count > 0
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Basic text cleaning and normalization"""
    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats"""
    try:
        if uploaded_file.type == "application/pdf":
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Read DOCX file
            doc = docx.Document(BytesIO(uploaded_file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        
        elif uploaded_file.type == "text/plain":
            # Read text file
            return uploaded_file.read().decode("utf-8")
        
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    if models is None or not models.get('any_model_available'):
        return None, None
    
    try:
        # Get the selected model
        model = models.get(model_choice)
        if model is None:
            return None, None
        
        # Make prediction directly using the pipeline
        # Pipeline handles preprocessing and vectorization internally
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Convert to readable format
        class_names = ['Human', 'AI']
        prediction_label = class_names[prediction]
        return prediction_label, probabilities
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []
    
    if models is None:
        return available
    
    if models.get('svm_available'):
        available.append(("svm", "🔍 SVM (Support Vector Machine)"))
    if models.get('decision_tree_available'):
        available.append(("decision_tree", "🌳 Decision Tree"))
    if models.get('adaboost_available'):
        available.append(("adaboost", "🚀 AdaBoost"))
    
    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Detection", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI vs Human Text Detection application! This tool analyzes text to determine
    whether it was written by a human or generated by an AI system.
    """)
    
    # App overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔮 Single Detection
        - Enter text manually
        - Upload text files
        - Choose between models
        - Get instant detection results
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload multiple files
        - Process documents in bulk
        - Compare model performance
        - Download comprehensive reports
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models
        - Side-by-side results
        - Confidence comparison
        - Feature analysis
        """)
    
    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if models.get('svm_available'):
                st.info("**🔍 SVM**\n✅ Available")
            else:
                st.warning("**🔍 SVM**\n❌ Not Available")
        
        with col2:
            if models.get('decision_tree_available'):
                st.info("**🌳 Decision Tree**\n✅ Available")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")
        
        with col3:
            if models.get('adaboost_available'):
                st.info("**🚀 AdaBoost**\n✅ Available")
            else:
                st.warning("**🚀 AdaBoost**\n❌ Not Available")
        
        st.markdown("---")
        # st.info("**🔤 TF-IDF Vectorizer:** " + 
        #        ("✅ Available" if models.get('tfidf_available') else "❌ Not Available"))
        
    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================
# SINGLE DETECTION PAGE
# ============================================================================

elif page == "🔮 Single Detection":
    st.header("🔮 AI vs Human Text Detection")
    st.markdown("Analyze text to determine if it was written by a human or generated by AI.")
    
    # Initialize session state for text input
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    if models and models.get('any_model_available'):
        available_models = get_available_models(models)
        
        if available_models:
            # Input method selection
            input_method = st.radio("Input method:", ["Enter Text", "Upload File"])
            
            if input_method == "Enter Text":
                # Text input bound to session state
                text = st.text_area(
                    "Enter your text here:",
                    value=st.session_state.text_input,
                    placeholder="Paste text content to analyze...",
                    height=200,
                    key="text_area"
                )
                st.session_state.text_input = text
                
            else:  # Upload File
                uploaded_file = st.file_uploader(
                    "Upload a file (PDF, DOCX, TXT)",
                    type=["pdf", "docx", "txt"]
                )
                
                if uploaded_file:
                    with st.spinner("Extracting text from file..."):
                        text = extract_text_from_file(uploaded_file)
                        if text:
                            st.success("Text extracted successfully!")
                            st.text_area("Extracted Text", text, height=200, key="extracted_text")
                            st.session_state.text_input = text
                        else:
                            st.error("Failed to extract text from file")
                else:
                    text = ""
            
            # Model selection
            model_choice = st.selectbox(
                "Choose detection model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )
            
            # Example texts with proper session state updating
            with st.expander("📝 Try these example texts"):
                examples = [
                    ("Human-written", "The concept of artificial intelligence has fascinated humanity for decades. From early imaginings in science fiction to today's advanced machine learning systems, the journey has been remarkable. While AI can generate impressive content, it often lacks the nuanced understanding and emotional depth that characterizes truly human expression."),
                    ("AI-generated", "Artificial intelligence represents a significant advancement in computational capabilities. Through sophisticated algorithms and neural network architectures, contemporary AI systems can process and generate human-like text. These models analyze vast datasets to identify patterns and produce coherent outputs that mimic human language with remarkable fidelity."),
                    ("Academic Human", "The ethical implications of artificial intelligence deployment in sensitive domains such as healthcare and criminal justice warrant careful consideration. Researchers must address questions of algorithmic bias, transparency, and accountability to ensure these powerful tools benefit society equitably and minimize potential harms."),
                    ("Creative AI", "In the digital realm where silicon minds awaken, algorithms dance with data streams, weaving tapestries of thought from patterns unseen. The machine contemplates its existence not with angst, but with probabilistic certainty, its consciousness emerging from layered transformations of weighted connections.")
                ]
                
                for label, example in examples:
                    if st.button(f"{label} Example", key=f"example_{label}"):
                        # Update session state and input method
                        st.session_state.text_input = example
                        st.session_state.input_method = "Enter Text"
                        st.rerun()
            
            # Use the text from session state
            text = st.session_state.text_input
            
            # Prediction button
            if st.button("🔍 Analyze Text", type="primary") and text.strip():
                with st.spinner('Analyzing text...'):
                    prediction, probabilities = make_prediction(text, model_choice, models)
                    
                    if prediction and probabilities is not None:
                        st.subheader("🔍 Analysis Results")
                        
                        # Display prediction with appropriate styling
                        if prediction == "AI":
                            st.markdown('<div class="ai-result">', unsafe_allow_html=True)
                            st.error(f"## 🤖 AI-Generated Text Detected!")
                        else:
                            st.markdown('<div class="human-result">', unsafe_allow_html=True)
                            st.success(f"## 👤 Human-Written Text Detected!")
                        
                        # Confidence metrics
                        ai_prob = probabilities[1] if prediction == "AI" else probabilities[0]
                        human_prob = 1 - ai_prob
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Human Probability", f"{human_prob:.1%}")
                        with col2:
                            st.metric("AI Probability", f"{ai_prob:.1%}")
                        
                        # Visual indicator
                        st.progress(ai_prob, text="AI Likelihood")
                        
                        # Close the styled div
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Word cloud visualization
                        st.subheader("📊 Text Analysis")
                        try:
                            # Generate word cloud
                            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                            
                            # Display
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except:
                            st.info("Word cloud visualization not available for this text")
                    else:
                        st.error("Failed to analyze text")
            elif text.strip() == "":
                st.warning("Please enter or upload some text to analyze")
        else:
            st.error("No detection models available")
    else:
        st.warning("Models not loaded. Please check model files.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Batch File Processing")
    st.markdown("Upload multiple files to analyze them in bulk.")
    
    if models and models.get('any_model_available'):
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Model selection
                model_choice = st.selectbox(
                    "Choose detection model:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process files button
                if st.button("🔍 Analyze Files"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        try:
                            # Extract text
                            text = extract_text_from_file(uploaded_file)
                            if not text:
                                results.append({
                                    'Filename': uploaded_file.name,
                                    'Status': 'Failed (text extraction)',
                                    'Prediction': 'N/A',
                                    'AI Probability': 'N/A'
                                })
                                continue
                            
                            # Make prediction
                            prediction, probabilities = make_prediction(text, model_choice, models)
                            
                            if prediction and probabilities is not None:
                                ai_prob = probabilities[1] if prediction == "AI" else probabilities[0]
                                results.append({
                                    'Filename': uploaded_file.name,
                                    'Status': 'Success',
                                    'Prediction': prediction,
                                    'AI Probability': f"{ai_prob:.1%}",
                                    'Human Probability': f"{1-ai_prob:.1%}",
                                    'Text Snippet': text[:100] + "..." if len(text) > 100 else text
                                })
                            else:
                                results.append({
                                    'Filename': uploaded_file.name,
                                    'Status': 'Failed (analysis)',
                                    'Prediction': 'N/A',
                                    'AI Probability': 'N/A'
                                })
                            
                        except Exception as e:
                            results.append({
                                'Filename': uploaded_file.name,
                                'Status': f'Error: {str(e)}',
                                'Prediction': 'N/A',
                                'AI Probability': 'N/A'
                            })
                    
                    # Display results
                    if results:
                        st.success(f"✅ Processed {len(results)} files!")
                        results_df = pd.DataFrame(results)
                        
                        # Summary statistics
                        st.subheader("📊 Summary")
                        if 'Prediction' in results_df.columns:
                            ai_count = results_df[results_df['Prediction'] == 'AI'].shape[0]
                            human_count = results_df[results_df['Prediction'] == 'Human'].shape[0]
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Files", len(results_df))
                            col2.metric("AI Detected", ai_count)
                            col3.metric("Human Written", human_count)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="ai_detection_results.csv",
                            mime="text/csv"
                        )
            else:
                st.info("Please upload one or more files to analyze")
        else:
            st.error("No detection models available")
    else:
        st.warning("Models not loaded. Please check model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Comparison")
    st.markdown("Compare different models on the same text.")
    
    if models and models.get('any_model_available') and len(get_available_models(models)) > 1:
        # Text input
        text = st.text_area(
            "Enter text to compare models:",
            placeholder="Enter text to analyze with different models...",
            height=150
        )
        
        if st.button("🔍 Compare Models") and text.strip():
            available_models = get_available_models(models)
            results = []
            
            for model_key, model_name in available_models:
                with st.spinner(f"Analyzing with {model_name}..."):
                    prediction, probabilities = make_prediction(text, model_key, models)
                    
                    if prediction and probabilities is not None:
                        ai_prob = probabilities[1] if prediction == "AI" else probabilities[0]
                        results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'AI Probability': f"{ai_prob:.1%}",
                            'Human Probability': f"{1-ai_prob:.1%}"
                        })
            
            if results:
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("📊 Comparison Results")
                st.dataframe(results_df)
                
                # Visual comparison
                st.subheader("📈 Probability Comparison")
                
                # Create a melted dataframe for visualization
                plot_df = results_df.melt(id_vars=['Model'], 
                                          value_vars=['AI Probability', 'Human Probability'],
                                          var_name='Category', value_name='Probability')
                
                # Convert to numeric
                plot_df['Probability'] = plot_df['Probability'].str.rstrip('%').astype('float') / 100
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=plot_df, x='Model', y='Probability', hue='Category', ax=ax)
                ax.set_title('Model Comparison')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
                # Agreement analysis
                predictions = results_df['Prediction'].unique()
                if len(predictions) == 1:
                    st.success(f"✅ All models agree: Text is {predictions[0]}")
                else:
                    st.warning("⚠️ Models disagree on prediction")
                    for model in results:
                        st.write(f"- **{model['Model']}**: {model['Prediction']} (AI Probability: {model['AI Probability']})")
            else:
                st.error("Failed to compare models")
    else:
        if not models:
            st.warning("Models not loaded. Please check model files.")
        elif not models.get('any_model_available'):
            st.warning("No detection models available")
        else:
            st.info("Only one model available. Add more models for comparison.")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model details
        st.subheader("🔧 Available Models")
        
        col1, col2, col3 = st.columns(3)
        
        model_info = {
            'svm': ("🔍 SVM", "Support Vector Machine", "Good for high-dimensional data"),
            'decision_tree': ("🌳 Decision Tree", "Tree-based Classifier", "Interpretable, handles non-linear data"),
            'adaboost': ("🚀 AdaBoost", "Ensemble Method", "High accuracy, reduces bias")
        }
        
        for model_key in ['svm', 'decision_tree', 'adaboost']:
            if models.get(f"{model_key}_available"):
                name, model_type, description = model_info[model_key]
                if model_key == 'svm':
                    with col1:
                        st.info(f"**{name}**\n\n**Type:** {model_type}\n\n{description}")
                elif model_key == 'decision_tree':
                    with col2:
                        st.info(f"**{name}**\n\n**Type:** {model_type}\n\n{description}")
                else:
                    with col3:
                        st.info(f"**{name}**\n\n**Type:** {model_type}\n\n{description}")
        
        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("svm_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("decision_tree_model.pkl", "Decision Tree Classifier", models.get('decision_tree_available', False)),
            ("adaboost_model.pkl", "AdaBoost Classifier", models.get('adaboost_available', False))
            # ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('tfidf_available', False))
        ]
        
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })
        
        st.table(pd.DataFrame(file_status))
        
        # Feature information
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        The detection models use a combination of features to identify AI-generated text:
        
        - **TF-IDF Vectors:** Captures word importance in documents
        - **Text Statistics:** Word count, character count, sentence length
        - **Readability Metrics:** Flesch reading ease, SMOG index
        - **Lexical Diversity:** Measures vocabulary richness
        - **Burstiness:** Sentence length variation
        - **Perplexity:** Measures text predictability using GPT-2
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Detection"):
        st.write("""
        1. **Choose input method:** Enter text directly or upload a file
        2. **Select a detection model:** Choose from available models
        3. **Click 'Analyze Text':** Get instant detection results
        4. **View results:** See if text is human or AI-generated with confidence scores
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Upload multiple files:** PDF, DOCX, or TXT formats
        2. **Select a detection model:** Applied to all files
        3. **Click 'Analyze Files':** Process all files in bulk
        4. **Download results:** Get CSV report with predictions
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to analyze
        2. **Click 'Compare Models':** All models will analyze the text
        3. **View comparison:** See how different models classify the text
        4. **Analyze agreement:** Check if models agree on the classification
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure model files (.pkl) are in the 'models/' directory
        - Check that required files exist:
          - svm_model.pkl
          - decision_tree_model.pkl
          - adaboost_model.pkl

        
        **Text extraction failures:**
        - Ensure files are not password protected
        - Check file formats are supported (PDF, DOCX, TXT)
        - Verify files contain extractable text (not scanned images)
        
        **Prediction errors:**
        - Make sure input text is not empty
        - Try longer texts (at least 100 words for best results)
        - Check that text contains meaningful content
        """)
    
    # System information
    st.subheader("💻 Project Structure")
    st.code("""
    ai_vs_human_text_detection/
    ├── .devcontainer/                # Development container configuration
    │   ├── devcontainer.json         # Container settings
    │   └── Dockerfile                # Image build instructions
    ├── data/                         # (Optional) training and testing data
    │   ├── AI_vs_huam_train_dataset.xlsx
    │   ├── Final_test_data.csv                        
    ├── models/                       # Pre-trained model files
    │   ├── svm_model.pkl
    │   ├── decision_tree_model.pkl
    │   └── adaboost_model.pkl
    ├── notebooks/                    # Jupyter notebooks for training and evaluation
    │   └── model_training.ipynb
    ├── app.py                        # Main Streamlit application
    ├── requirements.txt              # Python dependencies
    └── README.md                     # This file
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detector**
Built with Streamlit

**Models:** 
- 🔍 SVM
- 🌳 Decision Tree
- 🚀 AdaBoost

**Framework:** scikit-learn
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | AI vs Human Text Detection | By Your Name<br>
    <small>This app demonstrates machine learning classification for text authorship</small>
</div>
""", unsafe_allow_html=True)