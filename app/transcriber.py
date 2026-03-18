# from faster_whisper import WhisperModel

# # Modelo carregado uma vez só quando a aplicação sobe
# # Para começar localmente, CPU + int8 costuma ser mais leve
# model = WhisperModel(
#     "small",
#     device="cpu",
#     compute_type="int8"
# )


# def transcribe_audio(file_path: str) -> str:
#     segments, info = model.transcribe(
#         file_path,
#         beam_size=5
#     )

#     text_parts: list[str] = []

#     for segment in segments:
#         cleaned = segment.text.strip()
#         if cleaned:
#             text_parts.append(cleaned)

#     return " ".join(text_parts).strip()

from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

def transcribe_audio(file_path: str) -> str:
    segments, info = model.transcribe(
        file_path,
        beam_size=1,
        language="pt"
    )

    print("Detected language:", info.language)
    print("Language probability:", info.language_probability)

    text_parts = []

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        cleaned = segment.text.strip()
        if cleaned:
            text_parts.append(cleaned)

    return " ".join(text_parts).strip()