# Fake News Classification using LSTM

A deep learning-based fake news detection system deployed on Hugging Face Spaces. This project uses a sophisticated stacked LSTM architecture to classify news articles as real or fake with high accuracy.

🔗 **Live Demo**: [https://huggingface.co/spaces/Ashvitta07/Fake_news_classifier](https://huggingface.co/spaces/Ashvitta07/Fake_news_classifier)

## 🎯 Unique Features & Highlights

### 1. **Stacked LSTM Architecture**
- **Two-layer stacked LSTM** design (64 → 32 units) for enhanced sequential pattern learning
- Dropout regularization (0.2) at each LSTM layer to prevent overfitting
- Embedding layer with 100-dimensional vectors for rich semantic representation
- Achieved **~99% validation accuracy** on test data

### 2. **Advanced Text Preprocessing Pipeline**
- **Feature Fusion**: Combines article title and text content for richer context
- **Porter Stemming**: Reduces words to their root forms for better generalization
- **Stopword Removal**: Filters out common English stopwords
- **One-Hot Encoding**: Converts text to numerical representations (vocabulary size: 5000)
- **Sequence Padding**: Standardizes input length to 300 tokens with pre-padding

### 3. **Production-Ready Preprocessing Persistence**
- **Pickle-based Configuration**: Saves preprocessing parameters (`voc_size`, `sen_len`, `stop_words`, `stemmer`) in `preprocess.pkl`
- Ensures **consistent preprocessing** between training and inference
- Eliminates preprocessing discrepancies in production deployment

### 4. **Dockerized Deployment**
- **Optimized Dockerfile** for Hugging Face Spaces
- Includes system dependencies (git, git-lfs, ffmpeg, OpenGL libraries)
- Lightweight Python 3.10-slim base image
- Proper port configuration (7860) for Gradio interface

### 5. **Interactive Web Interface**
- **Gradio-based UI** with clean, user-friendly design
- **Visual Indicators**: 🟢 for real news, 🔴 for fake news
- **Confidence Scores**: Displays prediction probability for transparency
- Real-time inference with instant results

### 6. **Comprehensive Data Pipeline**
- **Separate Data Cleaning Notebook**: `Datasetcleaning.ipynb` for preprocessing workflow
- Balanced dataset handling (23,481 fake + 21,417 real news articles)
- Random shuffling with fixed seed for reproducibility
- Missing value handling and data validation

### 7. **Model Training Excellence**
- **Early Stopping Ready**: Architecture supports callback integration
- Binary cross-entropy loss with Adam optimizer
- Validation split monitoring (20% validation data)
- Batch size optimization (256) for efficient training

### 8. **Production Optimizations**
- **TensorFlow Logging Control**: Suppresses unnecessary warnings (`TF_CPP_MIN_LOG_LEVEL=2`)
- **NLTK Auto-download**: Handles stopwords download in Docker environment
- **Server Configuration**: Properly configured for containerized deployment (`0.0.0.0:7860`)

## 📊 Model Architecture

```
Input Text → Preprocessing → Embedding (5000 vocab, 100 dims)
    ↓
LSTM Layer 1 (64 units, dropout=0.2, return_sequences=True)
    ↓
LSTM Layer 2 (32 units, dropout=0.2)
    ↓
Dense Layer (1 unit, sigmoid activation)
    ↓
Output: Real (≥0.5) or Fake (<0.5)
```

## 📁 Project Structure

```
Fake-news-classification/
├── app.py                    # Gradio web application
├── Classification.ipynb     # Model training notebook
├── Datasetcleaning.ipynb     # Data preprocessing pipeline
├── fake_news_lstm.h5        # Trained LSTM model
├── preprocess.pkl           # Preprocessing configuration
├── fake_or_real_news.csv    # Processed dataset
├── Dockerfile               # Container configuration
└── requirements.txt         # Python dependencies
```

## 🚀 Technology Stack

- **Deep Learning**: TensorFlow/Keras 2.13.0
- **NLP**: NLTK (Porter Stemmer, Stopwords)
- **Web Framework**: Gradio 4.16.0
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Hugging Face Spaces

## 📈 Performance Metrics

- **Training Accuracy**: ~99.7%
- **Validation Accuracy**: ~99.0%
- **Dataset Size**: ~45,000 news articles
- **Vocabulary Size**: 5,000 words
- **Sequence Length**: 300 tokens

## 💡 Key Innovations

1. **Hybrid Feature Engineering**: Title + Text concatenation captures both headline impact and article depth
2. **Modular Preprocessing**: Pickle-based persistence ensures reproducibility across environments
3. **Stacked Architecture**: Multi-layer LSTM captures complex temporal dependencies
4. **Deployment Ready**: Complete Docker setup for seamless cloud deployment

## 🎓 Use Cases

- News verification systems
- Social media content moderation
- Educational tools for media literacy
- Research in misinformation detection

---

**Note**: This project demonstrates end-to-end ML pipeline from data cleaning to production deployment, showcasing best practices in deep learning for NLP tasks.
