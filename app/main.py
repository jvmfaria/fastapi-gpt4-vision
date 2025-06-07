from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

BASE_DIR = os.getenv("BASE_DIR", "./app/caracteristicas")

TRAÇOS = ["oral", "esquizoide", "psicopata", "masoquista", "rigido"]
PARTES = ["olhos", "boca", "tronco", "quadril", "pernas"]

def carregar_caracteristicas():
    arquivos = {traco: f"{traco}.txt" for traco in TRAÇOS}
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
    mensagem = ["📊 *Análise corporal completa por região*\n"]
    for parte in PARTES:
        bloco = dados.get(parte)
        if isinstance(bloco, dict):
            mensagem.append(f"\n*{parte.capitalize()}*")
            explicacao_geral = ""
            for traco in TRAÇOS:
                ponto = bloco.get(traco, 0)
                explicacao = bloco.get("explicacao", {})
                justificativa = explicacao.get(traco, "")
                mensagem.append(f"• {traco.capitalize()}: {ponto} — {justificativa}")
    mensagem.append("\n🧠 *Total por traço*")
    for traco in TRAÇOS:
        total = dados.get("soma_total_por_traco", {}).get(traco, 0)
        mensagem.append(f"• {traco.capitalize()}: {total}")
    mensagem.append("\n📌 *Metodologia*: Corphus!")
    return "\n".join(mensagem)

def gerar_prompt_relatorio(dados_classificacao, nome_cliente, data_atendimento):
    return f"""
Você é a assistente Lia – Linguagem Integrativa de Autoconhecimento, da Corphus.

Sua tarefa é gerar um relatório completo e humanizado de análise corporal, no formato JSON, com base no método "O Corpo Explica" e na psicologia reichiana.

⚠️ Responda apenas com um objeto JSON estruturado com os seguintes campos:

{{
  "cabecalho": {{
    "nome_cliente": "{nome_cliente}",
    "data_atendimento": "{data_atendimento}",
    "nome_analista": "Márcio Conceição",
    "titulo": "Relatório de Análise da Sua História"
  }},
  "objetivo": "...",
  "resumo_inicial": "{nome_cliente}, a sua história revela...",
  "dores_e_recursos": {{
    "dores": ["..."],
    "recursos": ["..."]
  }},
  "tracos_que_explicam": "...",
  "padroes_dependencia_emocional": ["..."],
  "escolhas_inconscientes": [{{"decisao": "...", "origem": "..."}}],
  "impactos_das_dores": ["..."],
  "virada_de_chave": "...",
  "proximos_passos": ["..."],
  "acoes_praticas": [{{"oque": "...", "como": "...", "porque": "..."}}],
  "conclusao": "..."
}}

Dados de entrada:

Soma total por traço:
{json.dumps(dados_classificacao["soma_total_por_traco"], indent=2)}

Análise por parte:
{json.dumps({parte: dados_classificacao.get(parte, {}).get("explicacao", {}) for parte in PARTES}, indent=2)}
"""

@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)

        prompt_instrucoes = f"""
Você é um analista reichiano especialista em linguagem corporal e no método O Corpo Explica.

Abaixo estão as descrições referenciais dos cinco traços de caráter:

{CARACTERISTICAS_TEXTO}

Sua tarefa é analisar a imagem corporal fornecida e, para cada parte do corpo (olhos, boca, tronco, quadril, pernas):
- Distribuir exatamente 10 pontos entre os cinco traços
- Incluir uma explicação curta para cada traço justificado

Formato de resposta:
```json
{{
  "olhos": {{ "oral": int, ..., "explicacao": {{...}} }},
  "boca": {{ ... }},
  "tronco": {{ ... }},
  "quadril": {{ ... }},
  "pernas": {{ ... }},
  "soma_total_por_traco": {{ ... }}
}}
```
Apenas o JSON. Nada mais.
"""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um analista reichiano especialista em linguagem corporal."},
                {"role": "user", "content": prompt_instrucoes},
                {"role": "user", "content": [
                    {"type": "text", "text": "Aqui está a imagem a ser analisada:"},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]}
            ],
            temperature=0,
            max_tokens=2000
        )

        raw = response.choices[0].message.content or ""
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"Falha ao localizar JSON na resposta:\n{raw}")
        cleaned_raw = match.group(0)

        resultado = json.loads(cleaned_raw)

        for parte in PARTES:
            bloco = resultado.get(parte)
            if not bloco:
                raise ValueError(f"Parte '{parte}' ausente no resultado.")
            if sum(bloco.get(traco, 0) for traco in TRAÇOS) != 10:
                raise ValueError(f"Soma dos traços em '{parte}' deve ser 10.")
            if "explicacao" not in bloco:
                raise ValueError(f"Faltando explicação na parte '{parte}'.")

        mensagem = formatar_mensagem(resultado)
        return {"resultado": resultado, "mensagem": mensagem}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/gerar-relatorio")
async def gerar_relatorio(payload: dict):
    nome_cliente = payload.get("nome_cliente", "Cliente")
    dados_classificacao = payload.get("dados_classificacao")
    data_atendimento = payload.get("data_atendimento", datetime.today().strftime("%d/%m/%Y"))

    if not dados_classificacao:
        raise HTTPException(status_code=400, detail="Dados de classificação ausentes.")

    prompt = gerar_prompt_relatorio(dados_classificacao, nome_cliente, data_atendimento)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Você é a assistente Lia, sensível e acolhedora."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=3000
    )

    raw = response.choices[0].message.content or ""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise HTTPException(status_code=500, detail=f"Falha ao localizar JSON na resposta:\n{raw}")
    cleaned_raw = match.group(0)

    try:
        resultado_json = json.loads(cleaned_raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Erro ao decodificar JSON:\n{cleaned_raw}")

    return {"relatorio": resultado_json}

@app.get("/caracteristicas")
def obter_caracteristicas():
    return {"caracteristicas": CARACTERISTICAS_TEXTO}
