from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente
load_dotenv()

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instância FastAPI
app = FastAPI()

# Diretório onde estão os arquivos dos traços de caráter
TRAITS_DIR = "D:/dataset_b/fastapi-gpt4-vision"

TRAITS = ["esquizoide", "masoquista", "oral", "psicopata", "rigido"]

# Função para converter imagem para base64 (data URL)
def file_to_data_url(file: UploadFile) -> str:
    content = file.file.read()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

# Função para carregar textos dos traços de caráter
def carregar_textos_tracos() -> str:
    textos = ""
    for traço in TRAITS:
        caminho = os.path.join(TRAITS_DIR, f"{traço}.txt")
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                textos += f"\n\n🔹 {traço.capitalize()}:\n{f.read()}"
    return textos.strip()

# Função principal de classificação
@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)
        textos_tracos = carregar_textos_tracos()

        prompt = (
            "Você é um analista experiente em psicologia corporal, especializado na análise reichiana. "
            "Seu papel é avaliar traços de caráter com base em uma imagem facial, utilizando os critérios fornecidos abaixo.\n\n"
            "Cada traço deve ser pontuado de 0 a 10, indicando o quanto ele está presente na expressão facial da pessoa. "
            "A soma total deve ser obrigatoriamente 10 pontos.\n\n"
            "📘 *Critérios para avaliação:*\n"
            f"{textos_tracos}\n\n"
            "📊 *Formato de resposta esperado (em JSON):*\n"
            "{\n"
            "  \"esquizoide\": <pontuação>,\n"
            "  \"masoquista\": <pontuação>,\n"
            "  \"oral\": <pontuação>,\n"
            "  \"psicopata\": <pontuação>,\n"
            "  \"rigido\": <pontuação>,\n"
            "  \"explicacao\": \"<explique de forma breve os principais indícios observados em cada traço com base na imagem>\"\n"
            "}"
        )

        response = client.chat.completions.create(
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
            max_tokens=1000
        )

        raw = response.choices[0].message.content

        try:
            # Remove blocos de código markdown, como ```json ... ```
            cleaned_raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
            resultado = json.loads(cleaned_raw)
            return resultado
        except json.JSONDecodeError:
            return {
                "erro": "A resposta não está em formato JSON válido.",
                "resposta_bruta": raw
            }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
