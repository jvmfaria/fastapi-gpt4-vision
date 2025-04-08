from fastapi import FastAPI, UploadFile, File
import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(file: UploadFile) -> str:
    image_data = file.file.read()
    return base64.b64encode(image_data).decode("utf-8")

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    image_base64 = encode_image(imagem)

    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em análise corporal reichiana. "
                    "Com base na imagem que será fornecida, classifique o tipo de caráter entre: "
                    "oral, esquizóide, masoquista, psicopata ou rígido. "
                    "Explique brevemente o porquê, baseado na análise da face humana."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise a imagem e diga qual o tipo de caráter segundo Reich:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=300
    )

    resposta = response.choices[0].message.content
    return {"tipo_carater": resposta}
