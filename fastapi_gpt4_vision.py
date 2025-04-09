from fastapi import FastAPI, UploadFile, File
import openai
import base64
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import tiktoken

# Carregar variáveis de ambiente
load_dotenv()

# Instanciar o FastAPI
app = FastAPI()

# Definir chave da OpenAI a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para codificar imagem em base64 com redimensionamento
def encode_image(file: UploadFile) -> str:
    image_data = file.file.read()
    # Usar o Pillow para redimensionar a imagem
    image = Image.open(BytesIO(image_data))
    image.thumbnail((800, 800))  # Ajuste o tamanho máximo desejado

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return image_base64

# Função para contar os tokens de uma string
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Endpoint de classificação de imagem
@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    # Codificar a imagem
    image_base64 = encode_image(imagem)

    # Prompt melhorado para o GPT-4 Vision
    prompt = (
        "Você é um especialista em análise psicológica reichiana com foco na classificação de tipos de caráter "
        "a partir da expressão facial. Com base na imagem fornecida, classifique o tipo de caráter da pessoa entre: "
        "oral, esquizóide, masoquista, psicopata ou rígido. Explique detalhadamente a razão de sua análise, "
        "levando em consideração as características faciais específicas que indicam esse tipo de caráter."
    )

    # Calcular os tokens do prompt + imagem
    total_tokens = count_tokens(prompt) + count_tokens(f"data:image/jpeg;base64,{image_base64}")

    # Garantir que o número de tokens não exceda o limite do modelo (8192 tokens para o GPT-4)
    if total_tokens > 8192:
        return {"erro": "O tamanho da entrada excede o limite de tokens permitido pelo modelo."}

    # Envio da imagem para o GPT-4V com o prompt
    response = openai.chat.completions.create(
        model="gpt-4",  # Usando o modelo correto GPT-4V
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"data:image/jpeg;base64,{image_base64}"}
        ],
        max_tokens=300
    )

    # Obter a resposta do modelo corretamente
    resposta = response.choice[0].message.content

    return {"tipo_carater": resposta}
