
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

def formatar_mensagem(dados):
    partes = ["olhos", "boca", "tronco", "quadril", "pernas"]
    mensagem = ["📊 *Análise corporal completa por região*\n"]
    for parte in partes:
        bloco = dados.get(parte)
        if isinstance(bloco, dict):
            mensagem.append(f"\n*{parte.capitalize()}*")
            for traco in ["oral", "esquizoide", "psicopata", "masoquista", "rigido"]:
                ponto = bloco.get(traco, 0)
                explicacao = bloco.get("explicacao", "")
                if isinstance(explicacao, dict):
                    justificativa = explicacao.get(traco, "")
                elif isinstance(explicacao, str):
                    justificativa = explicacao
                else:
                    justificativa = ""
                mensagem.append(f"• {traco.capitalize()}: {ponto} — {justificativa}")
    mensagem.append("\n🧠 *Total por traço*")
    for traco in ["oral", "esquizoide", "psicopata", "masoquista", "rigido"]:
        total = dados.get("soma_total_por_traco", {}).get(traco, 0)
        mensagem.append(f"• {traco.capitalize()}: {total}")
    mensagem.append("\n📌 *Metodologia*: O Corpo Explica")
    return "\n".join(mensagem)

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)

        prompt_instrucoes = (
            "Você é um analista reichiano e especialista no método O Corpo Explica.\n"
            "A seguir estão as características físicas e expressivas associadas a cinco traços de caráter:\n\n"
            f"{CARACTERISTICAS_TEXTO}\n"
            "Com base na imagem de corpo inteiro enviada, avalie separadamente as seguintes partes: olhos, boca, tronco, quadril e pernas.\n"
            "Para cada parte, atribua uma pontuação de 0 a 10 para cada um dos cinco traços de caráter:\n"
            "- Oral\n- Esquizoide\n- Psicopata\n- Masoquista\n- Rígido\n"
            "Em seguida, forneça uma explicação breve do que foi observado na imagem e como isso influenciou cada pontuação.\n"
            "No final, forneça a soma total por traço, considerando todas as partes.\n"
            "Responda exatamente no seguinte formato JSON:\n"
            "{\n"
            "  \"olhos\": {\n"
            "    \"oral\": 0-10, \"esquizoide\": 0-10, ..., \"explicacao\": { ... } ou \"<string>\"\n"
            "  },\n"
            "  \"boca\": { ... },\n"
            "  \"tronco\": { ... },\n"
            "  \"quadril\": { ... },\n"
            "  \"pernas\": { ... },\n"
            "  \"soma_total_por_traco\": {\n"
            "    \"oral\": <soma>, \"esquizoide\": <soma>, ...\n"
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
                        {"type": "text", "text": "Analise a imagem de corpo inteiro abaixo:"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            temperature=0,
            max_tokens=2000
        )

        raw = response.choices[0].message.content

        try:
            cleaned_raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
            resultado = json.loads(cleaned_raw)

            if not isinstance(resultado, dict):
                raise ValueError("Resposta JSON não é um dicionário.")

            mensagem = formatar_mensagem(resultado)
            return {
                "resultado": resultado,
                "mensagem": mensagem
            }
        except Exception as e:
            return {
                "erro": f"Erro ao interpretar resposta da OpenAI: {str(e)}",
                "resposta_bruta": raw
            }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/caracteristicas")
def obter_caracteristicas():
    return {"caracteristicas": CARACTERISTICAS_TEXTO}
