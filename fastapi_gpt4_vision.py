from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente do .env
load_dotenv()

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instância FastAPI
app = FastAPI()

# Diretório absoluto onde estão os arquivos dos traços de caráter
CARACTERES_DIR = "D:/dataset_b/fastapi-gpt4-vision"

# Converte imagem para base64 em formato data URL
def file_to_data_url(file: UploadFile) -> str:
    content = file.file.read()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

# Lê os arquivos TXT dos traços de caráter
def carregar_textos_dos_caracteres(diretorio=CARACTERES_DIR):
    tipos = ["esquizoide", "oral", "masoquista", "psicopata", "rigido"]
    textos = []
    for tipo in tipos:
        caminho = os.path.join(diretorio, f"{tipo}.txt")
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()
            textos.append(f"{tipo.capitalize()}:\n{texto.strip()}")
        else:
            textos.append(f"{tipo.capitalize()}:\n(Arquivo não encontrado)")
    return "\n\n".join(textos)

# Gera o prompt completo com base nos arquivos TXT
def gerar_prompt():
    tracos_texto = carregar_textos_dos_caracteres()
    prompt = (
        "Você é um analista reichiano experiente e também domina profundamente os conceitos do estudo 'O Corpo Explica'.\n\n"
        "A seguir estão os resumos dos cinco tipos de caráter com base nas observações físicas e comportamentais:\n\n"
        f"{tracos_texto}\n\n"
        "Com base na imagem facial fornecida, analise e classifique os traços da pessoa nos tipos de caráter: "
        "oral, esquizóide, masoquista, psicopata e rígido.\n\n"
        "Para cada tipo, atribua uma pontuação de 0 a 10 com base nas seguintes observações:\n"
        "- Formato da cabeça\n"
        "- Olhos\n"
        "- Boca\n"
        "- Postura\n"
        "- Expressões faciais\n\n"
        "A soma das pontuações deve ser sempre igual a 10.\n\n"
        "Retorne os resultados exclusivamente no seguinte formato JSON:\n"
        "{\n"
        "  \"oral\": <pontuação>,\n"
        "  \"esquizoide\": <pontuação>,\n"
        "  \"masoquista\": <pontuação>,\n"
        "  \"psicopata\": <pontuação>,\n"
        "  \"rigido\": <pontuação>,\n"
        "  \"explicacao\": \"<breve explicação do porquê de cada pontuação>\"\n"
        "}"
    )
    return prompt

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)
        prompt = gerar_prompt()

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
            # Remove blocos markdown ```json ... ``` se existirem
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
