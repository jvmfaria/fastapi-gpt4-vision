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

# Converte imagem para data URL base64
def file_to_data_url(file: UploadFile) -> str:
    content = file.file.read()
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded}"

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)

        prompt = (
            "Você é um analista reichiano experiente. Além disso, você também domina os conceitos do corpo explica."
            "Com base na imagem facial fornecida, analise e classifique os traços da pessoa nos tipos de caráter: "
            "oral, esquizóide, masoquista, psicopata e rígido. Para cada tipo, atribua uma pontuação de 0 a 10 indicando o quanto aquele traço está presente na expressão facial. "
            "A soma dos valores deve ser sempre 10"
            "Retorne os resultados exclusivamente no seguinte formato JSON:\n"
            "{\n"
            "  \"oral\": <pontuação>,\n"
            "  \"esquizoide\": <pontuação>,\n"
            "  \"masoquista\": <pontuação>,\n"
            "  \"psicopata\": <pontuação>,\n"
            "  \"rigido\": <pontuação>,\n"
            "  \"explicacao\": \"<breve explicação do porquê de cada pontuação>\"\n"
            "}\n"
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
            max_tokens=800
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
