from fastapi import FastAPI, UploadFile, File
import openai
import base64
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Instanciar o FastAPI
app = FastAPI()

# Definir chave da OpenAI a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para codificar imagem em base64
def encode_image(file: UploadFile) -> str:
    image_data = file.file.read()
    return base64.b64encode(image_data).decode("utf-8")

# Endpoint de classificação de imagem
@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    image_base64 = encode_image(imagem)

    # Prompt melhorado para o GPT-4 Vision
    prompt = (
        "Você é um especialista em análise psicológica reichiana com foco na classificação de tipos de caráter "
        "a partir da expressão facial. Com base na imagem fornecida, classifique o tipo de caráter da pessoa entre: "
        "oral, esquizóide, masoquista, psicopata ou rígido. Explique detalhadamente a razão de sua análise, "
        "levando em consideração as características faciais específicas que indicam esse tipo de caráter."
    )

    # Envio da imagem para o GPT-4 Vision com o prompt
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"data:image/jpeg;base64,{image_base64}"
            }
        ],
        max_tokens=300
    )

    # Obter a resposta do modelo
    resposta = response.choices[0].message["content"]
    return {"tipo_carater": resposta}
