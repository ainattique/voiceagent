import asyncio

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
)

SAMPLE_RATE = 24_000


def record_from_mic() -> AudioInput:
    print("Recording... press Enter to stop.")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    # FIX: Kept as a flat NumPy array so it retains the .dtype attribute
    audio_array = np.concatenate(chunks, axis=0).flatten()
    return AudioInput(buffer=audio_array)


agent = Agent(
    name="Travel Advisor",
    instructions=(
        "You are a friendly travel advisor. "
        "Keep spoken answers short, natural, and easy to follow."
        "Keep responses to 2-3 sentences max."
    ),
    model="gpt-4o-mini",
)


pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(agent),
    config=VoicePipelineConfig(
        tts_settings=TTSModelSettings(voice="coral"),
        trace_include_sensitive_audio_data=False,
    ),
)


async def main():
    input("Press Enter to start recording your question...")
    audio_input = record_from_mic()

    result = await pipeline.run(audio_input)

    player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    player.start()

    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)

    player.stop()
    player.close()

    print("\nText the agent generated:")
    print(result.total_output_text)


if __name__ == "__main__":
    asyncio.run(main())
