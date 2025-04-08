from fastapi import FastAPI, UploadFile, File
import openai

app = FastAPI()

# A chave será lida das variáveis de ambiente no Railway
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_VISION_MODEL = "gpt-4-vision-preview"

def processar_imagem_com_gpt(imagem: bytes):
    response = openai.ChatCompletion.create(
        model=GPT_VISION_MODEL,
        messages=[
            {"role": "system", "content": "Você é um especialista em análise de rostos baseada nos estudos de Wilhelm Reich."},
            {"role": "user", "content": "Analise essa foto e classifique a pessoa em um dos seguintes tipos de caráter: esquizóide, masoquista, oral, psicopata ou rígido, com base nos estudos de Wilhelm Reich. Justifique sua resposta."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + imagem.decode()}}]}
        ],
    )
    return response["choices"][0]["message"]["content"]

import base64

@app.post("/classificar")
async def classificar_imagem(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes)
    resultado = processar_imagem_com_gpt(image_base64)
    return {"classificacao": resultado}
