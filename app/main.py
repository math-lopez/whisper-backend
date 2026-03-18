import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from app.transcriber import transcribe_audio

app = FastAPI(title="Whisper Backend", version="1.0.0")

INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    x_internal_token: str | None = Header(default=None)
):
    if not INTERNAL_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="INTERNAL_API_TOKEN não configurado no servidor"
        )

    if x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido")

    allowed_extensions = {
        ".mp3", ".wav", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga", ".ogg"
    }

    _, extension = os.path.splitext(file.filename.lower())
    if extension and extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Extensão não suportada: {extension}"
        )

    temp_path = None

    try:
        suffix = extension if extension else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Arquivo vazio")

            tmp.write(contents)
            temp_path = tmp.name

        text = transcribe_audio(temp_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "text": text
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass