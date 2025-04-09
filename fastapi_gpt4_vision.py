from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import openai
import base64
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente (.env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Converte imagem para data URL base64
def file_to_data_url(file: UploadFile) -> str:
    content = file.file.read()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        # Converte imagem para data URL
        image_data_url = file_to_data_url(imagem)

        # Prompt especializado
        prompt = (
            "Você é um analista reichiano experiente. Com base na imagem facial fornecida, identifique e classifique "
            "o tipo de caráter da pessoa entre: oral, esquizóide, masoquista, psicopata ou rígido. Explique sua resposta "
            "citando as características visíveis que fundamentam sua análise psicológica."
        )

        # Chamada à OpenAI com GPT-4 Turbo (com suporte a visão)
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analise a imagem abaixo:"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            max_tokens=800
        )

        resposta = response.choices[0].message.content
        return {"tipo_carater": resposta}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
