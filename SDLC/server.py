from fastapi import FastAPI, WebSocket
import uvicorn
import numpy as np
import torch

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)

app = FastAPI()

print("Loading Silero...")

model = load_silero_vad()

print("Silero Loaded")

audio_buffer = []


@app.websocket("/audio")
async def audio_socket(websocket: WebSocket):

    global audio_buffer

    await websocket.accept()

    print("✅ Client Connected")

    try:

        while True:

            data = await websocket.receive_bytes()

            audio_chunk = np.frombuffer(
                data,
                dtype=np.float32
            )

            audio_buffer.extend(
                audio_chunk.tolist()
            )

            if len(audio_buffer) >= 16000 * 3:

                audio_tensor = torch.tensor(
                    audio_buffer,
                    dtype=torch.float32
                )

                speech = get_speech_timestamps(
                        audio_tensor,
                        model,
                        sampling_rate=16000,
                        return_seconds=True
                    )

                if speech:

                    print(
                        "🟢 SPEECH DETECTED"
                    )

                    for seg in speech:

                        print(
                            f"Start: {seg['start']:.2f}s | "
                            f"End: {seg['end']:.2f}s"
                        )

                else:

                    print(
                        "🔇 SILENCE"
                    )

                audio_buffer = []

    except Exception as e:

        print("ERROR:", e)


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )