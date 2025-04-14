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

# Diretório onde estão os arquivos TXT com os traços
CARACTER_DIR = "D:/dataset_b/fastapi-gpt4-vision"

# Função para ler os arquivos de traços
def carregar_conteudo_dos_tracos():
    tracos = ["esquizoide", "masoquista", "oral", "psicopata", "rigido"]
    conteudos = {}
    for traco in tracos:
        caminho = os.path.join(CARACTER_DIR, f"{traco}.txt")
        with open(caminho, "r", encoding="utf-8") as f:
            conteudos[traco] = f.read()
    return conteudos

# Converte imagem para base64
def file_to_data_url(file: UploadFile) -> str:
    content = file.file.read()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)
        conteudos_tracos = carregar_conteudo_dos_tracos()

        prompt = (
            "Você é um analista reichiano experiente e também domina os conceitos do 'O Corpo Explica'. "
            "A seguir estão descrições detalhadas de cada traço de caráter humano, organizadas por aspecto facial "
            "(formato da cabeça, olhos e boca). Use estas informações para analisar a imagem enviada e classificar os traços da pessoa.\n\n"
        )

        for traco, texto in conteudos_tracos.items():
            prompt += f"### Traço {traco.capitalize()}:\n{texto}\n\n"

        prompt += (
            "Com base na imagem facial, atribua uma pontuação de 0 a 10 para CADA ASPECTO (formato da cabeça, olhos e boca), "
            "distribuindo essa pontuação entre os 5 traços de caráter. A soma de cada aspecto deve ser exatamente 10 pontos.\n\n"
            "Formato da resposta esperado (em JSON):\n"
            "{\n"
            "  \"cabeca\": {\n"
            "    \"esquizoide\": <pontuação>,\n"
            "    \"masoquista\": <pontuação>,\n"
            "    \"oral\": <pontuação>,\n"
            "    \"psicopata\": <pontuação>,\n"
            "    \"rigido\": <pontuação>\n"
            "  },\n"
            "  \"olhos\": {\n"
            "    ...idem...\n"
            "  },\n"
            "  \"boca\": {\n"
            "    ...idem...\n"
            "  },\n"
            "  \"explicacao\": \"<breve explicação do porquê de cada pontuação>\"\n"
            "}\n"
        )

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
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
            # Remove blocos markdown tipo ```json
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
