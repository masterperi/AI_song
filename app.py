import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import os
from io import BytesIO
import base64

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
    }
    .audio-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the Granite model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

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
        ).to(model.device)
        
        # Generate with appropriate parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=min(len(text) * 2, 500),  # Adaptive length based on input
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
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

def create_audio_placeholder(text, voice):
    """Create a placeholder for audio generation (replace with actual TTS)"""
    # This is a placeholder function. In a real implementation, you would:
    # 1. Use IBM Watson Text-to-Speech API
    # 2. Generate actual audio file
    # 3. Return the audio file path or bytes
    
    audio_info = {
        "text_length": len(text),
        "voice": voice,
        "estimated_duration": len(text.split()) * 0.5,  # Rough estimate
        "format": "mp3"
    }
    
    return audio_info

def get_download_link(text, filename):
    """Generate download link for text file"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

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
        st.header("⚙ Configuration")
        
        # Model loading section
        st.subheader("Model Status")
        if not st.session_state.model_loaded:
            if st.button("🚀 Load Granite Model", type="primary"):
                with st.spinner("Loading AI model..."):
                    tokenizer, model, success = load_model()
                    if success:
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
        else:
            st.success("✅ Model ready")
        
        # Tone selection
        st.subheader("🎭 Tone Selection")
        tone = st.selectbox(
            "Choose narrative tone:",
            ["Neutral", "Suspenseful", "Inspiring"],
            help="Select how you want your text to be rewritten"
        )
        
        # Voice selection
        st.subheader("🎤 Voice Selection")
        voice = st.selectbox(
            "Choose narrator voice:",
            ["Lisa", "Michael", "Allison"],
            help="Select the voice for text-to-speech conversion"
        )
        
        # Tone descriptions
        st.subheader("📝 Tone Guide")
        tone_descriptions = {
            "Neutral": "Clear, professional, and objective narration",
            "Suspenseful": "Engaging tone that builds tension and interest",
            "Inspiring": "Motivational and uplifting delivery"
        }
        
        for t, desc in tone_descriptions.items():
            if t == tone:
                st.markdown(f"{t}: {desc}")
            else:
                st.markdown(f"{t}: {desc}")
    
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
            original_text = st.text_area(
                "Enter your text:",
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
                with st.spinner(f"Rewriting text with {tone} tone..."):
                    rewritten_text = rewrite_text_with_tone(
                        original_text, 
                        tone, 
                        st.session_state.tokenizer, 
                        st.session_state.model
                    )
                    st.session_state.rewritten_text = rewritten_text
                    st.session_state.current_tone = tone
            
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
                    get_download_link(
                        st.session_state.rewritten_text, 
                        f"rewritten_{st.session_state.current_tone.lower()}.txt"
                    ), 
                    unsafe_allow_html=True
                )
        
        elif not st.session_state.model_loaded:
            st.info("Please load the AI model first using the sidebar.")
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
            if st.button("🎧 Generate Audiobook", type="primary", use_container_width=True):
                with st.spinner(f"Generating audio with {voice} voice..."):
                    # Placeholder for audio generation
                    audio_info = create_audio_placeholder(st.session_state.rewritten_text, voice)
                    st.session_state.audio_info = audio_info
                    
                    # In a real implementation, you would generate actual audio here
                    st.success("✅ Audio generated successfully!")
                    st.info("📝 Note: This is a demo. In production, actual audio would be generated using IBM Watson Text-to-Speech.")
        
        with audio_col2:
            if hasattr(st.session_state, 'audio_info'):
                st.subheader("📊 Audio Details")
                info = st.session_state.audio_info
                st.write(f"*Voice:* {info['voice']}")
                st.write(f"*Duration:* ~{info['estimated_duration']:.1f}s")
                st.write(f"*Format:* {info['format'].upper()}")
                st.write(f"*Text Length:* {info['text_length']} chars")
        
        # Audio playback placeholder
        if hasattr(st.session_state, 'audio_info'):
            st.subheader("🎵 Audio Playback")
            st.info("🔊 Audio player would appear here in the full implementation")
            
            # Download button placeholder
            st.download_button(
                label="📥 Download MP3 (Demo)",
                data="This would be audio file bytes in real implementation",
                file_name=f"audiobook_{st.session_state.current_tone.lower()}_{voice.lower()}.mp3",
                mime="audio/mpeg",
                help="Download the generated audiobook"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎧 EchoVerse - Making content accessible through AI-powered audiobook creation</p>
        <p><small>Powered by IBM Granite 3.3 2B Instruct Model</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()