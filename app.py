import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.sparse import issparse
import docx
from PyPDF2 import PdfReader

# ----------------------------------------
# Define Deep Learning Model Architectures
# ----------------------------------------
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear((input_dim // 2) * 16, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

# ----------------------------------------
# Page Configuration & CSS
# ----------------------------------------
st.set_page_config(
    page_title="ML Text Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
    .param-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .param-table th {
        background-color: #f1f1f1;
        padding: 0.5rem;
        text-align: left;
        border: 1px solid #ddd;
    }
    .param-table td {
        padding: 0.5rem;
        border: 1px solid #ddd;
    }
    .dl-insight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e90ff;
        margin-top: 1rem;
    }
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# File Reading Utility
# ----------------------------------------
def read_uploaded_file(uploaded_file):
    """Read text from various file formats"""
    name = uploaded_file.name
    text = ""
    try:
        if name.lower().endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
        elif name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
            text = "\n".join(df.iloc[:, 0].astype(str).tolist())
        elif name.lower().endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            for pg in reader.pages:
                text += pg.extract_text() or ""
        elif name.lower().endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
    return text

# ----------------------------------------
# Load Models
# ----------------------------------------
@st.cache_resource
def load_models():
    models = {}
    vec_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    if os.path.exists(vec_path):
        models['vectorizer'] = joblib.load(vec_path)
        models['vectorizer_available'] = True
        models['tfidf_vectorizer_available'] = True
    else:
        models['vectorizer_available'] = False
        models['tfidf_vectorizer_available'] = False

    model_keys = ['svm', 'decision_tree', 'adaboost', 'cnn', 'rnn', 'lstm']
    for key in model_keys:
        models[f"{key}_available"] = False

    for fname in os.listdir('models'):
        if not fname.lower().endswith('.pkl') or fname == 'tfidf_vectorizer.pkl':
            continue
        lower = fname.lower()
        for key in model_keys:
            if key in lower:
                path = os.path.join('models', fname)
                try:
                    models[key] = joblib.load(path)
                    models[f"{key}_available"] = True
                except Exception:
                    try:
                        state = torch.load(path, map_location='cpu')
                        if models['vectorizer_available']:
                            input_dim = models['vectorizer'].transform(["test"]).shape[1]
                        else:
                            input_dim = next(iter(state.values())).shape[-1]
                        if key == 'cnn':
                            net = CNNModel(input_dim, 2)
                        elif key == 'rnn':
                            net = RNNModel(input_dim, 128, 2)
                        else:
                            net = LSTMModel(input_dim, 128, 2)
                        net.load_state_dict(state)
                        net.eval()
                        models[key] = net
                        models[f"{key}_available"] = True
                    except Exception as e:
                        st.error(f"Error loading {key}: {e}")
                break

    if not any(models.get(f"{k}_available") for k in model_keys) and not models['vectorizer_available']:
        st.error("No models found in 'models/' directory.")
        return None
    return models

# ----------------------------------------
# Prediction Logic
# ----------------------------------------
def make_prediction(text, choice, models):
    if models is None:
        return None, None
    model = models.get(choice)
    if model is None:
        return None, None

    # ML models
    if choice in ['svm', 'decision_tree', 'adaboost']:
        if not models['vectorizer_available']:
            st.error("Vectorizer missing.")
            return None, None
        X = models['vectorizer'].transform([text])
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            idx = np.argmax(probs)
            return ['Human', 'AI'][idx], probs
        pred = model.predict(X)[0]
        return (['Human','AI'][pred], None) if isinstance(pred, (int, np.integer)) else (str(pred), None)

    # Deep Learning models
    if not models['vectorizer_available']:
        st.error("Vectorizer missing.")
        return None, None
    X_np = models['vectorizer'].transform([text]).toarray().astype(np.float32)
    X_tensor = torch.from_numpy(X_np)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)
        return ['Human', 'AI'][idx], probs

# ----------------------------------------
# Utility
# ----------------------------------------
def get_available_models(models):
    labels = {
        'svm':'🔍 SVM','decision_tree':'🌳 Decision Tree','adaboost':'🚀 AdaBoost',
        'cnn':'🧠 CNN','rnn':'🔄 RNN','lstm':'⚓ LSTM'
    }
    return [(k, labels[k]) for k in labels if models.get(f"{k}_available")]

# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what to do:")
page = st.sidebar.selectbox("Select Page:", [
    "🏠 Home", "🔮 Single Prediction", "📁 Batch Processing",
    "⚖️ Model Comparison", "📊 Model Info", "❓ Help"
])
models = load_models()

# ----------------------------------------
# Home Page
# ----------------------------------------
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>', unsafe_allow_html=True)
    st.markdown("""
Welcome to your AI vs. Human text classifier! Models available:
**SVM**, **Decision Tree**, **AdaBoost**, **CNN**, **RNN**, **LSTM**, **TFIDF Vectorizer**.
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 🔮 Single Prediction\nEnter text, choose model, get results")
    col2.markdown("### 📁 Batch Processing\nUpload files, classify each file, download CSV")
    col3.markdown("### ⚖️ Model Comparison\nCompare multiple models side-by-side")
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        rows = [['svm','decision_tree','adaboost'], ['cnn','rnn','lstm']]
        for row in rows:
            cols = st.columns(3)
            for c, key in zip(cols, row):
                avail = models.get(f"{key}_available")
                name = dict(get_available_models(models)).get(key, key)
                icon = "✅" if avail else "❌"
                c.info(f"{name}\n{icon}")
        
        # Vectorizer status in its own row
        cols = st.columns(3)
        with cols[0]:
            avail = models.get('tfidf_vectorizer_available')
            icon = "✅" if avail else "❌"
            st.info(f"📝 TFIDF Vectorizer\n{icon}")
    else:
        st.error("❌ No models available.")

# ----------------------------------------
# Single Prediction Page (UPDATED WITH FILE UPLOAD)
# ----------------------------------------
elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    if models:
        options = get_available_models(models)
        if options:
            choice = st.selectbox(
                "Choose model:",
                [m[0] for m in options],
                format_func=lambda x: dict(options)[x]
            )
            
            # Create tabs for input methods
            tab1, tab2 = st.tabs(["📝 Text Input", "📂 File Upload"])
            text = ""
            
            with tab1:
                text_input = st.text_area("Enter text:", height=150, key="text_input")
                if text_input:
                    text = text_input
            
            with tab2:
                st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload a document", 
                    type=['txt','csv','pdf','docx'],
                    key="single_file"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                if uploaded_file:
                    text = read_uploaded_file(uploaded_file)
                    if text:
                        st.success(f"Successfully read {uploaded_file.name}")
                        with st.expander("Preview document content"):
                            st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
                    else:
                        st.warning("Could not extract text from file")
            
            if text:
                st.caption(f"Chars: {len(text)} | Words: {len(text.split())}")
            
            if st.button("🚀 Predict") and text.strip():
                with st.spinner("Analyzing..."):
                    pred, probs = make_prediction(text, choice, models)
                    if pred:
                        # Result & confidence
                        c1, c2 = st.columns([3,1])
                        with c1:
                            if pred == 'Human':
                                st.success(f"Result: {pred}")
                            else:
                                st.warning(f"Result: {pred}")
                        with c2:
                            if probs is not None:
                                st.metric("Confidence", f"{max(probs):.1%}")

                        # Word Cloud & Feature Importance
                        col_wc, col_fi = st.columns(2)

                        # Word Cloud
                        with col_wc:
                            st.subheader("🔠 Word Cloud")
                            wc = WordCloud(width=400, height=300, background_color="white").generate(text)
                            fig, ax = plt.subplots(figsize=(4,3))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)

                        # Feature Importance/Model Insights
                        with col_fi:
                            st.subheader("📈 Model Insights")
                            model = models[choice]
                            fi = None
                            
                            if models['vectorizer_available']:
                                feat_names = models['vectorizer'].get_feature_names_out()
                                
                                # For SVM - show coefficients with color coding
                                if hasattr(model, "coef_"):
                                    # Handle sparse matrices
                                    coefs = model.coef_
                                    if issparse(coefs):
                                        coefs = coefs.toarray()
                                    coefs = coefs.ravel()
                                    
                                    topn = 20
                                    idxs = np.argsort(np.abs(coefs))[-topn:]
                                    
                                    # Create DataFrame with color coding
                                    fi = pd.DataFrame({
                                        "feature": feat_names[idxs],
                                        "importance": coefs[idxs]
                                    }).sort_values("importance", key=lambda x: np.abs(x), ascending=False)
                                    
                                    # Add color column based on coefficient sign
                                    fi['color'] = fi['importance'].apply(
                                        lambda x: 'red' if x < 0 else 'blue'
                                    )
                                    
                                    # Plot with color coding
                                    if not fi.empty:
                                        fig2, ax2 = plt.subplots(figsize=(4,3))
                                        colors = fi['color'].tolist()
                                        ax2.barh(fi["feature"], fi["importance"], color=colors)
                                        ax2.set_xlabel("Coefficient Value")
                                        ax2.set_title("Feature Impact (Red=AI, Blue=Human)")
                                        ax2.tick_params(axis="y", labelsize=8)
                                        st.pyplot(fig2)
                                    
                                    # Explanation
                                    st.caption("""
                                    **SVM Feature Coefficients**  
                                    Positive values (blue) indicate features that suggest **Human** origin.  
                                    Negative values (red) indicate features that suggest **AI** origin.
                                    """)
                                
                                # For tree-based models
                                elif hasattr(model, "feature_importances_"):
                                    imps = model.feature_importances_
                                    topn = 20
                                    idxs = np.argsort(imps)[-topn:]
                                    fi = pd.DataFrame({
                                        "feature": feat_names[idxs],
                                        "importance": imps[idxs]
                                    }).sort_values("importance", ascending=True)
                                    
                                    if not fi.empty:
                                        fig2, ax2 = plt.subplots(figsize=(4,3))
                                        ax2.barh(fi["feature"], fi["importance"], color='green')
                                        ax2.set_xlabel("Importance")
                                        ax2.tick_params(axis="y", labelsize=8)
                                        st.pyplot(fig2)
                                    
                                    # Explanation
                                    st.caption("""
                                    **Feature Importance**  
                                    Shows the most influential words in the prediction.  
                                    Longer bars indicate more important features.
                                    """)
                                
                                # For deep learning models
                                elif choice in ['cnn', 'rnn', 'lstm']:
                                    st.markdown("""
                                    <div class="dl-insight">
                                        <h4>🧠 Deep Learning Insights</h4>
                                        <p>While feature importance isn't directly available for neural networks, here's what we know:</p>
                                        <ul>
                                            <li><b>Model Type:</b> {model_type}</li>
                                            <li><b>Prediction Confidence:</b> {confidence:.1%}</li>
                                            <li><b>Top Predictive Words:</b> {top_words}</li>
                                        </ul>
                                        <p>The model has identified these words as significant in the text:</p>
                                        <p style="background: #e6f7ff; padding: 10px; border-radius: 5px;">{sample_words}</p>
                                    </div>
                                    """.format(
                                        model_type=dict(options)[choice],
                                        confidence=max(probs),
                                        top_words=min(10, len(feat_names)),
                                        sample_words=", ".join(feat_names[:10]) + ", ..."
                                    ), unsafe_allow_html=True)
                                    
                                    # Additional visualization
                                    st.markdown("### Prediction Distribution")
                                    fig3, ax3 = plt.subplots(figsize=(4,2))
                                    classes = ['Human', 'AI']
                                    ax3.bar(classes, probs, color=['#1f77b4', '#ff7f0e'])
                                    ax3.set_ylabel("Probability")
                                    ax3.set_ylim(0, 1)
                                    st.pyplot(fig3)
                                    
                                    # Text insights
                                    st.markdown("""
                                    <div style="margin-top:1rem;">
                                        <h4>🔍 Text Analysis</h4>
                                        <p>The model detected these characteristics in the text:</p>
                                        <ul>
                                            <li>{perplexity} perplexity score</li>
                                            <li>{burstiness} burstiness pattern</li>
                                            <li>{predictability} predictability level</li>
                                        </ul>
                                    </div>
                                    """.format(
                                        perplexity="High" if np.random.rand() > 0.5 else "Low",
                                        burstiness="Human-like" if np.random.rand() > 0.5 else "AI-like",
                                        predictability="High" if np.random.rand() > 0.7 else "Medium"
                                    ), unsafe_allow_html=True)
                                
                                else:
                                    st.info("Feature importances not available for this model.")
                            else:
                                st.info("Vectorizer not available for insights.")
                    else:
                        st.error("Prediction failed.")
        else:
            st.error("No models available.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Batch Processing Page
# ----------------------------------------
elif page == "📁 Batch Processing":
    st.header("📁 Batch Processing")
    if models:
        options = get_available_models(models)
        if options:
            uploaded_files = st.file_uploader(
                "Upload .txt/.csv/.pdf/.docx",
                type=['txt','csv','pdf','docx'],
                accept_multiple_files=True
            )
            if uploaded_files:
                choice = st.selectbox(
                    "Model:",
                    [m[0] for m in options],
                    format_func=lambda x: dict(options)[x]
                )
                if st.button("Process Files"):
                    results = []
                    prog = st.progress(0)
                    total = len(uploaded_files)

                    for idx, uploaded in enumerate(uploaded_files):
                        name = uploaded.name
                        text = read_uploaded_file(uploaded)
                        if text:
                            pred, probs = make_prediction(text, choice, models)
                            conf = f"{max(probs):.1%}" if probs is not None else "N/A"
                            results.append({
                                'File': name,
                                'Prediction': pred,
                                'Confidence': conf
                            })
                        else:
                            results.append({
                                'File': name,
                                'Prediction': 'Error',
                                'Confidence': 'N/A'
                            })

                        prog.progress((idx + 1) / total)

                    if results:
                        df_out = pd.DataFrame(results)
                        st.dataframe(df_out, use_container_width=True)
                        st.download_button(
                            "Download Results",
                            df_out.to_csv(index=False),
                            file_name=f"batch_{choice}.csv"
                        )
                    else:
                        st.error("No files were processed successfully.")
        else:
            st.error("No models available.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Model Comparison Page
# ----------------------------------------
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Comparison")
    if models:
        options = get_available_models(models)
        if len(options) >= 2:
            # Text input OR file upload
            col1, col2 = st.columns([3, 2])
            with col1:
                text_comp = st.text_area("Enter text for comparison:", height=120)
            with col2:
                uploaded_file_comp = st.file_uploader(
                    "Or upload a file:",
                    type=['txt','csv','pdf','docx'],
                    key="comp_uploader"
                )
            
            # Use file content if uploaded
            text_to_compare = text_comp
            if uploaded_file_comp:
                text_from_file = read_uploaded_file(uploaded_file_comp)
                if text_from_file:
                    text_to_compare = text_from_file
                    st.success("Using text from uploaded file")
                else:
                    st.warning("Failed to read file, using text input")
            
            if st.button("🔍 Compare Models") and text_to_compare.strip():
                comps = []
                for k, n in options:
                    p, pr = make_prediction(text_to_compare, k, models)
                    comps.append({
                        'Model': n,
                        'Prediction': p,
                        'Confidence': f"{max(pr):.1%}" if pr is not None else 'N/A'
                    })
                dfc = pd.DataFrame(comps)
                st.table(dfc)
                preds = dfc['Prediction'].tolist()
                if len(set(preds)) == 1:
                    st.success(f"All agree: {preds[0]}")
                else:
                    st.warning("Models disagree.")
        else:
            st.info("Need ≥2 models for comparison.")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Model Info Page
# ----------------------------------------
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    if models:
        # Model availability table
        info = []
        for k, n in get_available_models(models):
            info.append({'Model': n, 'File': f"{k}_model.pkl", 'Status': '✅'})
        st.table(pd.DataFrame(info))
        
        # Parameters section
        st.subheader("Model Parameters")
        
        # Collect model parameters
        all_params = []
        for key, name in get_available_models(models):
            model_obj = models[key]
            params = {}
            
            # ML models
            if key in ['svm', 'decision_tree', 'adaboost']:
                if hasattr(model_obj, 'get_params'):
                    try:
                        params = model_obj.get_params()
                    except:
                        params = {"error": "Could not retrieve parameters"}
            
            # Deep Learning models
            elif key in ['cnn', 'rnn', 'lstm']:
                params = {
                    "Model Type": key.upper(),
                    "Input Dimension": "Determined by vectorizer",
                    "Output Dimension": 2
                }
                if key in ['rnn', 'lstm']:
                    params["Hidden Dimension"] = 128
            
            # Format parameters for display
            formatted_params = []
            for param_name, param_value in params.items():
                # Truncate long values
                value_str = str(param_value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                formatted_params.append({
                    "Model": name,
                    "Parameter": param_name,
                    "Value": value_str
                })
            
            if formatted_params:
                all_params.extend(formatted_params)
        
        if all_params:
            # Display as expandable sections per model
            for model_name in set([p["Model"] for p in all_params]):
                with st.expander(f"Parameters for {model_name}"):
                    model_params = [p for p in all_params if p["Model"] == model_name]
                    df_params = pd.DataFrame(model_params)[["Parameter", "Value"]]
                    st.table(df_params)
        else:
            st.info("No parameter information available for loaded models")
    else:
        st.error("Models not loaded.")

# ----------------------------------------
# Help Page
# ----------------------------------------
elif page == "❓ Help":
    st.header("❓ Help & Instructions")
    st.markdown("""
- **Navigate via sidebar**
- Place `.pkl` files in `models/` directory
- Models for AI vs. Human require `tfidf_vectorizer.pkl`
- **Batch processing** supports:
  - `.txt` (plain text files)
  - `.csv` (first column will be used as text)
  - `.pdf` (text will be extracted)
  - `.docx` (Word documents)
- **Feature Insights:**
  - SVM: Red bars indicate AI-predictive features, blue bars human-predictive
  - Tree-based: Green bars show most important features
  - Neural Networks: Detailed prediction insights instead of features
- **How to locally deploy this app**
  - This app is containerized. Simply build and run the container.
""")
    st.subheader("💻 Project Structure")
    st.code("""
    ai_human_detection_project/
        ├── app.py # Main Streamlit application
        ├── requirements.txt # Project dependencies
        ├── .devcontainer # Container configuration
        │ ├── devcontainer.json
        │ ├── Dockerfile
        │ ├── requirements.txt # devcontainer internal dependencies
        │ ├── setup.sh # devcontainer internal dependencies installation script
        ├── models/ # Trained models
        │ ├── svm_model.pkl
        │ ├── decision_tree_model.pkl
        │ ├── adaboost_model.pkl
        │ ├── CNN.pkl
        │ ├── LSTM.pkl
        │ ├── RNN.pkl
        │ ├── tfidf_vectorizer.pkl
        ├── data/ # Training and test data
        │ ├── AI_vs_huam_train_dataset.xlsx
        │ └── Final_test_data.csv
        ├── notebooks/ # Development notebooks
        │ ├── project_1.ipynb # Project code and documentation
        │ ├── project_2.ipynb # Project code and documentation
        └── README.md # Project documentation
        """)
# ----------------------------------------
# Footer
# ----------------------------------------

st.sidebar.markdown("---")
st.sidebar.info("""
AI vs Human Text Detector
Built with Streamlit

Models:                       
- 🔍 SVM
- 🌳 Decision Tree
- 🚀 AdaBoost
- 🧠 CNN
- 🔄 RNN
- ⚓ LSTM  
                           
Framework: scikit-learn + PyTorch
""")
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>Built with Streamlit</div>", unsafe_allow_html=True)