from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

BASE_DIR = os.getenv("BASE_DIR", "./app/caracteristicas")

def carregar_caracteristicas():
    arquivos = {
        "oral": "oral.txt",
        "esquizoide": "esquizoide.txt",
        "masoquista": "masoquista.txt",
        "psicopata": "psicopata.txt",
        "rigido": "rigido.txt"
    }
    conteudos = []
    for traco, nome_arquivo in arquivos.items():
        caminho = os.path.join(BASE_DIR, nome_arquivo)
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                conteudos.append(f"{traco.upper()}:\n{f.read().strip()}\n")
    return "\n".join(conteudos)

CARACTERISTICAS_TEXTO = carregar_caracteristicas()

def file_to_data_url(file: UploadFile) -> str:
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")
    content = file.file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Arquivo de imagem excede 5MB.")
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)

        prompt_instrucoes = (
            "A seguir estão as características físicas e expressivas associadas a cinco traços de caráter:\n\n"
            f"{CARACTERISTICAS_TEXTO}\n"
            "Com base na imagem facial enviada, avalie a presença de cada traço de caráter e forneça uma pontuação de 0 a 10, "
            "sendo que a soma total deve ser igual a 10.\n"
            "Para cada traço, forneça uma explicação breve do que foi observado na imagem e como isso influenciou sua pontuação.\n"
            "Retorne no seguinte formato JSON:\n"
            "{\n"
            "  \"oral\": <pontuação>,\n"
            "  \"esquizoide\": <pontuação>,\n"
            "  \"masoquista\": <pontuação>,\n"
            "  \"psicopata\": <pontuação>,\n"
            "  \"rigido\": <pontuação>,\n"
            "  \"explicacao\": {\n"
            "    \"oral\": \"<justificativa>\",\n"
            "    \"esquizoide\": \"<justificativa>\",\n"
            "    \"masoquista\": \"<justificativa>\",\n"
            "    \"psicopata\": \"<justificativa>\",\n"
            "    \"rigido\": \"<justificativa>\"\n"
            "  }\n"
            "}\n"
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um analista reichiano e especialista no método O Corpo Explica."},
                {"role": "user", "content": prompt_instrucoes},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analise a imagem abaixo:"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            temperature=0,
            max_tokens=1000
        )

        raw = response.choices[0].message.content

        try:
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

@app.get("/caracteristicas")
def obter_caracteristicas():
    return {"caracteristicas": CARACTERISTICAS_TEXTO}