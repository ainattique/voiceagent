import asyncio
import hashlib
import io
import os
import wave

import numpy as np
import streamlit as st
from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
)

SAMPLE_RATE = 24_000

st.set_page_config(page_title="Travel Voice Agent", page_icon="🎙️")
st.title("🎙️ Travel Advisor Voice Agent")
st.write("Record a question and receive a spoken travel-advice response.")


def get_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OPENAI_API_KEY")


def decode_wav(wav_bytes):
    """Decode microphone WAV data into mono int16 audio at 24 kHz."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if width != 2:
            raise ValueError("Expected 16-bit PCM microphone audio.")
        audio = np.frombuffer(frames, dtype="<i2")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        if audio.size == 0:
            raise ValueError("The recording is empty.")

        if source_rate != SAMPLE_RATE:
            new_size = max(1, round(len(audio) * SAMPLE_RATE / source_rate))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_size),
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)

        return np.ascontiguousarray(audio, dtype=np.int16)
    except wave.Error as exc:
        raise ValueError("The microphone recording could not be decoded as WAV.") from exc


def make_pipeline(voice):
    agent = Agent(
        name="Travel Advisor",
        instructions=(
            "You are a friendly, knowledgeable travel advisor. Help users plan trips, "
            "compare destinations, and create practical itineraries. Ask a brief "
            "follow-up if essential details are missing. Keep spoken replies natural "
            "and concise, usually 2–4 sentences. Never invent live prices, visa rules, "
            "opening hours, or availability; say when details need verification."
        ),
        model="gpt-4o-mini",
    )
    return VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        config=VoicePipelineConfig(
            tts_settings=TTSModelSettings(voice=voice),
            trace_include_sensitive_audio_data=False,
        ),
    )


async def process_voice_turn(audio_array, voice):
    pipeline = make_pipeline(voice)
    audio_input = AudioInput(
        buffer=audio_array,
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )
    result = await pipeline.run(audio_input)

    chunks = []
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            chunks.append(np.asarray(event.data).copy())

    if not chunks:
        raise RuntimeError("The agent did not return any audio.")
    output = np.concatenate(chunks)
    if output.dtype == np.float32:
        output = (np.clip(output, -1, 1) * 32767).astype(np.int16)
    else:
        output = output.astype(np.int16, copy=False)
    return result.total_output_text, output


def to_wav_bytes(audio):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(np.asarray(audio, dtype="<i2").tobytes())
    return buffer.getvalue()


api_key = get_api_key()
if not api_key:
    st.warning(
        'Set OPENAI_API_KEY in Streamlit Cloud → App settings → Secrets, for example: '
        'OPENAI_API_KEY = "your-api-key". Locally, set it as an environment variable.'
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

with st.sidebar:
    st.subheader("Voice settings")
    voice = st.selectbox(
        "Assistant voice",
        ["coral", "marin", "cedar", "alloy", "ash", "ballad", "echo",
         "fable", "nova", "onyx", "sage", "shimmer", "verse"],
        index=0,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/wav")

recording = st.audio_input("Record your travel question", sample_rate=SAMPLE_RATE)

if recording is not None:
    raw_audio = recording.getvalue()
    recording_id = hashlib.sha256(raw_audio).hexdigest()

    if st.session_state.get("processed_recording") != recording_id:
        if st.button("Send to Travel Advisor", type="primary"):
            try:
                st.session_state.processed_recording = recording_id
                with st.spinner("The agent is listening and preparing your answer..."):
                    audio_array = decode_wav(raw_audio)
                    answer, spoken_audio = asyncio.run(
                        process_voice_turn(audio_array, voice)
                    )
                    answer_wav = to_wav_bytes(spoken_audio)

                st.session_state.messages.append(
                    {"role": "user", "text": "🎤 Voice question"}
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": answer or "I could not generate a text answer.",
                        "audio": answer_wav,
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Voice agent error: {exc}")
                st.info("Check your API key, network connection, and installed dependencies.")

if st.session_state.messages and st.button("Clear conversation"):
    st.session_state.messages = []
    st.session_state.pop("processed_recording", None)
    st.rerun()

st.divider()
st.caption(
    "Audio is sent to OpenAI for transcription and response generation. "
    "Do not record passwords, payment details, or other sensitive information."
)
