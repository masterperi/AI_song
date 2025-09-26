import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import os
from io import BytesIO
import base64
import pyttsx3
import threading
from gtts import gTTS
import pygame
import time

# Page configuration
st.set_page_config(
    page_title="EchoVerse - AI Audiobook Creator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .tone-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .text-comparison {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #111; /* <-- Add this line for black text */
    }
    .audio-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .audio-player {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_loaded = False
    st.session_state.tts_engine = None
    st.session_state.audio_generated = False

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init()
except:
    pass

@st.cache_resource
def load_model():
    """Load the Granite model with memory optimization"""
    try:
        # Memory optimization settings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        tokenizer = AutoTokenizer.from_pretrained(
            "ibm-granite/granite-3.3-2b-instruct",
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True      # Load model with less memory usage
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-2b-instruct",
            torch_dtype=torch.float16,  # Half precision
            low_cpu_mem_usage=True,     # Memory efficient loading
            device_map="auto"           # Automatically handle device placement
        )
        
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Error loading Granite model: {str(e)}")
        st.info("Try increasing your virtual memory (paging file) in Windows settings")
        return None, None, False

@st.cache_resource
def initialize_tts_engine():
    """Initialize the local TTS engine"""
    try:
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        
        # Set default properties
        engine.setProperty('rate', 180)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        return engine, voices
    except Exception as e:
        st.error(f"Error initializing TTS engine: {str(e)}")
        return None, None

def rewrite_text_with_tone(text, tone, tokenizer, model):
    """Rewrite text with specified tone using Granite model"""
    
    # Define tone-specific prompts
    tone_prompts = {
        "Neutral": f"Rewrite the following text in a clear, neutral, and professional tone while preserving all the original meaning and information:\n\n{text}",
        "Suspenseful": f"Rewrite the following text with a suspenseful and engaging tone that creates tension and interest while maintaining the original meaning:\n\n{text}",
        "Inspiring": f"Rewrite the following text with an inspiring and motivational tone that uplifts the reader while preserving the original meaning:\n\n{text}"
    }
    
    try:
        messages = [
            {"role": "user", "content": tone_prompts[tone]},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with memory optimization
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=min(len(text) * 2, 400),  # Reduced for memory
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable caching for efficiency
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        ).strip()
        
        return generated_text
        
    except Exception as e:
        st.error(f"Error rewriting text: {str(e)}")
        return text

def generate_audio_pyttsx3(text, voice_id, rate=180):
    """Generate audio using pyttsx3 (offline TTS)"""
    try:
        if st.session_state.tts_engine is None:
            engine, voices = initialize_tts_engine()
            st.session_state.tts_engine = engine
            st.session_state.available_voices = voices
        
        engine = st.session_state.tts_engine
        voices = st.session_state.available_voices
        
        # Set voice
        if voices and voice_id < len(voices):
            engine.setProperty('voice', voices[voice_id].id)
        
        # Set rate
        engine.setProperty('rate', rate)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio to file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        return temp_path
    except Exception as e:
        st.error(f"Error generating audio with pyttsx3: {str(e)}")
        return None

def generate_audio_gtts(text, lang='en', slow=False):
    """Generate audio using gTTS (requires internet for first use)"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio to file
        tts.save(temp_path)
        
        return temp_path
    except Exception as e:
        st.error(f"Error generating audio with gTTS: {str(e)}")
        st.info("Note: gTTS requires internet connection. Using offline pyttsx3 instead.")
        return None

def get_download_link(file_path, filename):
    """Generate download link for audio file"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return ""

def get_text_download_link(text, filename):
    """Generate download link for text file"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📄 Download {filename}</a>'
    return href

# Sample texts
sample_texts = {
    "Educational": "Photosynthesis is the process by which plants convert sunlight into energy. During this remarkable biological process, plants absorb carbon dioxide from the atmosphere and water from their roots. Using chlorophyll as a catalyst, they transform these simple ingredients into glucose and oxygen. This process not only feeds the plant but also produces the oxygen that most life on Earth depends upon.",
    
    "Mystery": "Detective Martinez examined the crime scene with careful precision. The antique clock had stopped at exactly midnight, and a single red rose lay on the mahogany desk. Three suspects had alibis, but something didn't add up. As she studied the evidence, a peculiar detail caught her attention - a barely visible footprint in the dust that shouldn't have been there.",
    
    "Motivational": "Success is not about avoiding failure, but about learning from every setback. Each challenge you face is an opportunity to grow stronger and wiser. Remember that every champion was once a beginner who refused to give up. Your dreams are valid, your efforts matter, and your persistence will pay off.",
    
    "Business": "Effective teamwork requires clear communication, shared goals, and mutual respect. In today's global marketplace, diverse teams bring unique perspectives that drive innovation. Companies that foster collaborative environments see improved productivity and employee satisfaction."
}

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎧 EchoVerse</h1>
        <p>Transform your text into expressive audiobooks with AI-powered tone adaptation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading section
        st.subheader("🚀 Model Setup")
        if not st.session_state.model_loaded:
            st.info("IBM Granite 3.3 2B Instruct Model (~4-5GB)")
            st.warning("⚠️ Large model - ensure 8GB+ RAM available")
            if st.button("🚀 Load Granite Model", type="primary"):
                with st.spinner("Loading IBM Granite model... (5-10 minutes)"):
                    tokenizer, model, success = load_model()
                    if success:
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.success("✅ Granite model loaded successfully!")
                        st.balloons()
                        st.rerun()
        else:
            st.success("✅ Granite Model Ready")
        
        # TTS Engine selection
        st.subheader("🎤 Text-to-Speech Engine")
        tts_engine = st.radio(
            "Choose TTS engine:",
            ["pyttsx3 (Offline)", "gTTS (Online)"],
            help="pyttsx3 works offline but has limited voices. gTTS requires internet but sounds more natural."
        )
        
        # Tone selection
        st.subheader("🎭 Tone Selection")
        tone = st.selectbox(
            "Choose narrative tone:",
            ["Neutral", "Suspenseful", "Inspiring"],
            help="Select how you want your text to be rewritten"
        )
        
        # Voice/Speed settings
        st.subheader("🎛️ Audio Settings")
        if tts_engine == "pyttsx3 (Offline)":
            # Initialize TTS to get available voices
            if st.session_state.tts_engine is None:
                engine, voices = initialize_tts_engine()
                st.session_state.tts_engine = engine
                st.session_state.available_voices = voices
            
            voices = st.session_state.available_voices if st.session_state.available_voices else []
            voice_names = [f"Voice {i}: {v.name[:30]}" for i, v in enumerate(voices)] if voices else ["Default Voice"]
            
            voice_index = st.selectbox("Select voice:", range(len(voice_names)), format_func=lambda x: voice_names[x])
            speech_rate = st.slider("Speech rate:", 100, 300, 180, help="Words per minute")
        else:
            language = st.selectbox("Language:", ["en", "en-us", "en-uk", "en-au"], index=0)
            slow_speech = st.checkbox("Slow speech", value=False)
        
        # Tone descriptions
        st.subheader("📝 Tone Guide")
        tone_descriptions = {
            "Neutral": "Clear, professional, and objective narration",
            "Suspenseful": "Engaging tone that builds tension and interest",
            "Inspiring": "Motivational and uplifting delivery"
        }
        
        for t, desc in tone_descriptions.items():
            if t == tone:
                st.markdown(f"**{t}**: {desc}")
            else:
                st.markdown(f"{t}: {desc}")
    
    # Sample text selection
    st.header("📝 Sample Texts")
    
    selected_sample = st.selectbox("Choose a sample text:", ["Custom"] + list(sample_texts.keys()))
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Text input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"],
            horizontal=True
        )
        
        original_text = ""
        
        if input_method == "Paste Text":
            if selected_sample != "Custom":
                default_text = sample_texts[selected_sample]
            else:
                default_text = ""
                
            original_text = st.text_area(
                "Enter your text:",
                value=default_text,
                height=300,
                placeholder="Paste your text here...",
                help="Enter the text you want to convert to audiobook"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt'],
                help="Upload a .txt file containing your text"
            )
            
            if uploaded_file is not None:
                original_text = str(uploaded_file.read(), "utf-8")
                st.text_area(
                    "Uploaded text:",
                    value=original_text,
                    height=300,
                    disabled=True
                )
    
    with col2:
        st.header("✨ Tone-Adapted Text")
        
        if original_text and st.session_state.model_loaded:
            if st.button("🔄 Rewrite with Selected Tone", type="primary"):
                with st.spinner(f"Rewriting text with {tone} tone using Granite model..."):
                    rewritten_text = rewrite_text_with_tone(
                        original_text, 
                        tone, 
                        st.session_state.tokenizer, 
                        st.session_state.model
                    )
                    st.session_state.rewritten_text = rewritten_text
                    st.session_state.current_tone = tone
                    st.session_state.audio_generated = False  # Reset audio flag
            
            # Display rewritten text
            if hasattr(st.session_state, 'rewritten_text'):
                st.text_area(
                    f"Text rewritten with {st.session_state.current_tone} tone:",
                    value=st.session_state.rewritten_text,
                    height=300,
                    help="This is your text adapted to the selected tone"
                )
                
                # Download option for rewritten text
                st.markdown(
                    get_text_download_link(
                        st.session_state.rewritten_text, 
                        f"rewritten_{st.session_state.current_tone.lower()}.txt"
                    ), 
                    unsafe_allow_html=True
                )
        
        elif not st.session_state.model_loaded:
            st.info("Please load the Granite model first using the sidebar.")
        else:
            st.info("Please enter some text to rewrite.")
    
    # Text comparison section
    if original_text and hasattr(st.session_state, 'rewritten_text'):
        st.header("🔄 Side-by-Side Comparison")
        
        comp_col1, comp_col2 = st.columns([1, 1])
        
        with comp_col1:
            st.subheader("Original Text")
            st.markdown(f'<div class="text-comparison">{original_text}</div>', 
                       unsafe_allow_html=True)
        
        with comp_col2:
            st.subheader(f"{st.session_state.current_tone} Tone")
            st.markdown(f'<div class="text-comparison">{st.session_state.rewritten_text}</div>', 
                       unsafe_allow_html=True)
    
    # Audio generation section
    if hasattr(st.session_state, 'rewritten_text'):
        st.markdown('<div class="audio-section">', unsafe_allow_html=True)
        st.header("🎵 Audio Generation")
        
        audio_col1, audio_col2 = st.columns([2, 1])
        
        with audio_col1:
            st.subheader("📊 Text Analysis")
            text = st.session_state.rewritten_text
            word_count = len(text.split())
            char_count = len(text)
            
            if tts_engine == "pyttsx3 (Offline)":
                estimated_duration = word_count / (speech_rate / 60)  # Based on WPM
            else:
                estimated_duration = word_count * 0.5  # Rough estimate
            
            st.write(f"**TTS Engine:** {tts_engine}")
            st.write(f"**Tone:** {st.session_state.current_tone}")
            st.write(f"**Word Count:** {word_count} words")
            st.write(f"**Character Count:** {char_count} characters")
            st.write(f"**Estimated Duration:** ~{estimated_duration:.1f} seconds ({estimated_duration/60:.1f} minutes)")
            
            # Generate audio button
            if st.button("🎤 Generate Audio", type="primary"):
                with st.spinner("Generating audio... This may take a moment..."):
                    if tts_engine == "pyttsx3 (Offline)":
                        audio_path = generate_audio_pyttsx3(text, voice_index, speech_rate)
                    else:
                        audio_path = generate_audio_gtts(text, language, slow_speech)
                    
                    if audio_path:
                        st.session_state.audio_path = audio_path
                        st.session_state.audio_generated = True
                        st.success("✅ Audio generated successfully!")
                    else:
                        st.error("❌ Failed to generate audio. Please try again.")
        
        with audio_col2:
            st.subheader("🎛️ Current Settings")
            
            if tts_engine == "pyttsx3 (Offline)":
                st.write(f"**Voice:** {voice_names[voice_index] if 'voice_names' in locals() else 'Default'}")
                st.write(f"**Speech Rate:** {speech_rate} WPM")
            else:
                st.write(f"**Language:** {language}")
                st.write(f"**Slow Speech:** {'Yes' if slow_speech else 'No'}")
        
        # Audio player section
        if hasattr(st.session_state, 'audio_generated') and st.session_state.audio_generated:
            st.markdown('<div class="audio-player">', unsafe_allow_html=True)
            st.subheader("🎧 Generated Audiobook")
            
            try:
                # Display audio player
                with open(st.session_state.audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav' if tts_engine == "pyttsx3 (Offline)" else 'audio/mp3')
                
                # Download button
                filename = f"audiobook_{st.session_state.current_tone.lower()}.{'wav' if tts_engine == 'pyttsx3 (Offline)' else 'mp3'}"
                st.markdown(
                    get_download_link(st.session_state.audio_path, filename),
                    unsafe_allow_html=True
                )
                
                # File info
                file_size = os.path.getsize(st.session_state.audio_path) / 1024 / 1024  # MB
                st.write(f"**File size:** {file_size:.2f} MB")
                
            except Exception as e:
                st.error(f"Error displaying audio: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Installation guide
    with st.expander("🛠️ Installation Requirements"):
        st.markdown("""
        **Required packages for local TTS:**
        
        ```bash
        pip install pyttsx3 gtts pygame
        ```
        
        **For pyttsx3 (offline TTS):**
        - Works completely offline
        - Uses system TTS voices
        - Fast generation
        - Platform-specific voices (Windows SAPI, macOS NSSpeechSynthesizer, Linux espeak)
        
        **For gTTS (Google TTS):**
        - Requires internet connection
        - High-quality, natural voices
        - Supports multiple languages
        - Slower generation due to API calls
        
        **System Requirements:**
        - Python 3.7+
        - For GPU acceleration: CUDA-compatible GPU with 4GB+ VRAM
        - RAM: 8GB+ recommended for Granite model
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎧 EchoVerse - Making content accessible through AI-powered audiobook creation</p>
        <p><small>Powered by IBM Granite 3.3 2B Instruct Model + Local Text-to-Speech</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
