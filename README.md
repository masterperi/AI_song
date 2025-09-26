# 🎧 EchoVerse - AI Audiobook Creator

Transform your text into expressive audiobooks with AI-powered tone adaptation and local text-to-speech generation.

![EchoVerse Demo](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🌟 Features

### 🤖 **AI-Powered Text Processing**
- **IBM Granite 3.3 2B Instruct Model** for intelligent text rewriting
- **Tone Adaptation**: Transform text into Neutral, Suspenseful, or Inspiring tones
- **Context-Aware Rewriting**: Maintains original meaning while adapting style

### 🎤 **Local Text-to-Speech**
- **Dual TTS Engines**: 
  - `pyttsx3` - Completely offline, uses system voices
  - `gTTS` - Online, high-quality Google voices
- **Voice Customization**: Multiple voice options and speech rate control
- **No External APIs**: Generate audio without cloud dependencies

### 🎵 **Audio Generation**
- **Real-time Audio Creation**: Generate audiobooks instantly
- **Built-in Audio Player**: Play generated audio directly in the app
- **Downloadable Files**: Save both rewritten text and audio files
- **Multiple Formats**: Support for WAV and MP3 audio formats

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- 8GB+ RAM (recommended for Granite model)
- GPU with 4GB+ VRAM (optional, for faster processing)

### Installation

1. **Install required packages:**
```bash
pip install streamlit torch transformers pyttsx3 gtts pygame numpy
```

2. **Run the application:**
```bash
streamlit run app.py
```

3. **Access the app:**
   - Open your browser and go to `http://localhost:8501`

## 📦 Dependencies

Create a `requirements.txt` file with:
```txt
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
pyttsx3>=2.90
gtts>=2.3.0
pygame>=2.5.0
numpy>=1.24.0
```

## 🎯 Usage Guide

### 1. **Load the AI Model**
- Click "🚀 Load Granite Model" in the sidebar
- Wait 5-10 minutes for the IBM Granite 3.3 2B model to load
- Model is cached for faster subsequent loads

### 2. **Configure Settings**
- **TTS Engine**: Choose between offline (pyttsx3) or online (gTTS)
- **Tone**: Select Neutral, Suspenseful, or Inspiring
- **Voice Settings**: Adjust speech rate and voice selection

### 3. **Input Text**
- **Paste Text**: Direct text input
- **Upload File**: Upload .txt files
- **Sample Texts**: Use pre-built examples

### 4. **Generate Content**
- Click "🔄 Rewrite with Selected Tone"
- Review the AI-adapted text
- Click "🎤 Generate Audio" to create audiobook

### 5. **Download & Share**
- Play audio directly in the browser
- Download rewritten text as .txt file
- Download audio as WAV/MP3 file

## 🛠️ Configuration Options

### TTS Engine Settings

| Engine | Pros | Cons | Best For |
|--------|------|------|----------|
| **pyttsx3** | Offline, Fast, No limits | Basic voices | Privacy, Speed |
| **gTTS** | Natural voices, Multi-language | Requires internet | Quality, Languages |

### Tone Options

| Tone | Description | Use Case |
|------|-------------|----------|
| **Neutral** | Clear, professional narration | Business, Educational |
| **Suspenseful** | Engaging, tension-building | Mystery, Thrillers |
| **Inspiring** | Motivational, uplifting | Self-help, Speeches |

## 📁 Project Structure

```
echoverse/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── temp/                 # Temporary audio files (auto-created)
└── models/               # Cached models (auto-created)
```

## 🔧 Advanced Configuration

### GPU Acceleration
The app automatically detects and uses GPU if available:
```python
# Automatic GPU detection
device_map="auto"
torch_dtype=torch.float16  # Half precision for memory efficiency
```

### Memory Optimization
For systems with limited RAM:
- Model uses half precision (float16)
- Low CPU memory usage mode
- Automatic memory cleanup

### Custom Voice Installation

**Windows:**
- Install additional SAPI voices from Microsoft Store
- Use Speech Platform SDK for more voice options

**macOS:**
- System Preferences > Accessibility > Speech
- Download additional voices

**Linux:**
- Install espeak-data for more voices: `sudo apt-get install espeak espeak-data`

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Increase virtual memory on Windows
# Control Panel > System > Advanced > Performance Settings > Advanced > Virtual Memory
```

**Audio Generation Issues:**
```bash
# For Linux users
sudo apt-get update
sudo apt-get install alsa-utils pulseaudio
```

**Memory Issues:**
- Close other applications
- Use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`
- Reduce max_new_tokens in the model generation

### Performance Tips

1. **First-time model download**: The Granite model (~4-5GB) downloads automatically
2. **Subsequent runs**: Model loads from cache (much faster)
3. **Audio generation**: pyttsx3 is faster, gTTS has better quality
4. **GPU usage**: Automatically detects and uses GPU if available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Granite**: For the powerful 3.3 2B Instruct model
- **Streamlit**: For the amazing web framework
- **Transformers**: For easy model integration
- **pyttsx3 & gTTS**: For text-to-speech capabilities

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration options

---

**Made with ❤️ for audiobook enthusiasts and accessibility advocates**
