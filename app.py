import asyncio
import os
import queue
import sys
import numpy as np
import sounddevice as sd
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Ensure layout config is first
st.set_page_config(
    page_title="VibeVoyage AI | Voice Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

try:
    from agents import Agent
    from agents.voice import (
        AudioInput,
        SingleAgentVoiceWorkflow,
        TTSModelSettings,
        VoicePipeline,
        VoicePipelineConfig,
    )
except ImportError:
    st.error("❌ Critical components from the `agents` library are missing! Please check installation.")

# Constants
SAMPLE_RATE = 24_000

# Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "agent_status" not in st.session_state:
    st.session_state.agent_status = "Ready to assist"

# Setup the backend Voice Pipeline
@st.cache_resource
def get_voice_pipeline():
    agent = Agent(
        name="Travel Advisor",
        instructions=(
            "You are a friendly, enthusiastic, and knowledgeable travel advisor. "
            "Keep spoken answers short, completely natural, and highly engaging. "
            "Keep responses to a strict limit of 2-3 sentences maximum."
        ),
        model="gpt-4o-mini",
    )
    return VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        config=VoicePipelineConfig(
            tts_settings=TTSModelSettings(voice="coral"),
            trace_include_sensitive_audio_data=False,
        ),
    )

try:
    pipeline = get_voice_pipeline()
except Exception as e:
    st.sidebar.error(f"Error configuring pipeline: {e}")

# Title Section
st.title("✈️ VibeVoyage AI")
st.subheader("Your Intelligent Agentic Voice Companion")
st.markdown("---")

# Layout Configuration
col_control, col_display = st.columns([1, 2], gap="large")

with col_control:
    st.header("🎛️ Voice Control Panel")
    st.info("💡 Click **Start Recording**, speak your travel question clearly, and click **Stop & Process** when you're finished.")
    
    # State-driven interactive loop button actions
    if not st.session_state.is_recording:
        if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.agent_status = "🔴 Recording microphone input..."
            st.session_state.chunks = []
            
            # Start background non-blocking audio stream
            def callback(indata, frames, time, status):
                if st.session_state.is_recording:
                    st.session_state.chunks.append(indata.copy())
                    
            st.session_state.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype="int16", 
                callback=callback
            )
            st.session_state.stream.start()
            st.rerun()
    else:
        if st.button("🛑 Stop & Process", type="secondary", use_container_width=True):
            st.session_state.is_recording = False
            st.session_state.agent_status = "🧠 Agentic AI processing audio..."
            st.rerun()

    # Status Monitor
    st.metric(label="System Status", value=st.session_state.agent_status)
    
    # Audio Processing & Execution Loop
    if not st.session_state.is_recording and hasattr(st.session_state, "chunks") and len(st.session_state.chunks) > 0:
        with st.spinner("Processing voice request and synthesizing response stream..."):
            try:
                # Stop existing hardware audio input streams safely
                if hasattr(st.session_state, "stream"):
                    st.session_state.stream.stop()
                    st.session_state.stream.close()
                
                # Reconstruct and validate vector shape properties for AudioInput
                audio_array = np.concatenate(st.session_state.chunks, axis=0).flatten()
                audio_input = AudioInput(buffer=audio_array)
                st.session_state.chunks = [] # Clear immediately to reset context
                
                # Execute Pipeline asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(pipeline.run(audio_input))
                
                # Local variables to track final text output elements
                generated_text = result.total_output_text if hasattr(result, 'total_output_text') else ""
                
                # Process structural playback buffer events
                audio_chunks = []
                async def collect_stream():
                    async for event in result.stream():
                        if event.type == "voice_stream_event_audio":
                            audio_chunks.append(event.data)
                loop.run_until_complete(collect_stream())
                
                # Streamlit clean native audio component player rendering
                if audio_chunks:
                    full_audio_bytes = b"".join(audio_chunks)
                    st.audio(full_audio_bytes, format="audio/wav", autoplay=True)
                    st.success("🔊 Playing back personalized response!")
                
                # Persist conversational chat nodes in application session states
                st.session_state.chat_history.append({"role": "user", "text": "🎤 [Voice Request]"})
                st.session_state.chat_history.append({"role": "assistant", "text": generated_text})
                st.session_state.agent_status = "Ready to assist"
                st.rerun()
                
            except Exception as ex:
                st.error(f"Pipeline processing failed: {ex}")
                st.session_state.agent_status = "Error state"

with col_display:
    st.header("💬 Conversation Transcript")
    
    # Render updated interaction loops chronologically
    if not st.session_state.chat_history:
        st.write("*No conversation history yet. Start recording to consult your advisor.*")
    else:
        for index, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.chat_message("user").write(message["text"])
            else:
                st.chat_message("assistant").write(message["text"])

    if st.button("🗑️ Clear Chat Sessions", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()