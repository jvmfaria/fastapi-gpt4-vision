from fastapi import FastAPI, UploadFile, File
import openai
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

# Carregar variáveis de ambiente
load_dotenv()

# Instanciar o FastAPI
app = FastAPI()

# Definir chave da OpenAI a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para codificar imagem em base64
def encode_image(file: UploadFile) -> str:
    # Abrir a imagem utilizando PIL e transformar para o formato correto
    image_data = file.file.read()
    image = Image.open(io.BytesIO(image_data))

    # Aqui, podemos realizar qualquer manipulação de imagem, se necessário
    # Exemplo: redimensionar a imagem
    image = image.convert("RGB")  # Garantir que a imagem esteja no formato RGB
    image_data = io.BytesIO()
    image.save(image_data, format="JPEG")
    image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")

    return image_base64

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

    try:
        # Envio da imagem para o GPT-4 Vision com o prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Usar o modelo GPT-4 Vision
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
        resposta = response.choices[0].message.content
        return {"tipo_carater": resposta}

    except Exception as e:
        return {"error": str(e)}
