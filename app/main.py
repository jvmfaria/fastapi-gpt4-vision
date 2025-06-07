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
    mensagem = ["\U0001F4CA *Análise corporal completa por região*\n"]
    for parte in PARTES:
        bloco = dados.get(parte)
        if isinstance(bloco, dict):
            mensagem.append(f"\n*{parte.capitalize()}*")
            explicacao_geral = ""
            for traco in TRAÇOS:
                ponto = bloco.get(traco, 0)
                explicacao = bloco.get("explicacao", "")
                if isinstance(explicacao, dict):
                    justificativa = explicacao.get(traco, "")
                    mensagem.append(f"• {traco.capitalize()}: {ponto} — {justificativa}")
                elif isinstance(explicacao, str):
                    explicacao_geral = explicacao
                    mensagem.append(f"• {traco.capitalize()}: {ponto}")
            if explicacao_geral:
                mensagem.append(f"\U0001F50E Observação: {explicacao_geral}")
    mensagem.append("\n\U0001F9E0 *Total por traço*")
    for traco in TRAÇOS:
        total = dados.get("soma_total_por_traco", {}).get(traco, 0)
        mensagem.append(f"• {traco.capitalize()}: {total}")
    mensagem.append("\n\U0001F4CC *Metodologia*: Corphus!")
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
  "objetivo": "texto explicando o objetivo do relatório",
  "resumo_inicial": "texto acolhedor começando com '{nome_cliente}, a sua história revela...'",
  "dores_e_recursos": {{
    "dores": ["listar 3 a 5 dores reais"],
    "recursos": ["listar 3 a 5 forças internas"]
  }},
  "tracos_que_explicam": "texto explicando os traços dominantes e como atuam",
  "padroes_dependencia_emocional": ["listar ao menos 3 padrões com exemplo"],
  "escolhas_inconscientes": [
    {{
      "decisao": "descrição da decisão inconsciente",
      "origem": "qual medo ou dor originou essa decisão"
    }}
  ],
  "impactos_das_dores": ["3 a 5 impactos na vida conjugal, financeira, familiar"],
  "virada_de_chave": "nova forma de pensar e agir proposta",
  "proximos_passos": ["3 a 5 decisões estratégicas"],
  "acoes_praticas": [
    {{
      "oque": "ação prática",
      "como": "como realizar",
      "porque": "propósito emocional por trás"
    }}
  ],
  "conclusao": "mensagem final motivacional"
}}

Dados de entrada para análise:

Soma total por traço:
{json.dumps(dados_classificacao["soma_total_por_traco"], indent=2)}

Análise por parte do corpo:
{json.dumps({parte: dados_classificacao[parte]["explicacao"] for parte in PARTES}, indent=2)}

⚠️ Retorne somente um objeto JSON válido. Não inclua explicações fora dele.
"""


@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)

        prompt_instrucoes = f"""
Você é um analista reichiano e especialista no método O Corpo Explica.
A seguir estão as descrições de referência dos cinco traços de caráter:

{CARACTERISTICAS_TEXTO}

Com base na imagem de corpo inteiro enviada, avalie separadamente as seguintes partes do corpo:
- Olhos
- Boca
- Tronco
- Quadril
- Pernas

Distribua exatamente 10 pontos entre os cinco traços para cada parte do corpo.

⚠️ IMPORTANTE: Sua resposta deve ser obrigatoriamente no formato JSON válido, com a seguinte estrutura:

{{
  "olhos": {{...}},
  "boca": {{...}},
  "tronco": {{...}},
  "quadril": {{...}},
  "pernas": {{...}},
  "soma_total_por_traco": {{
    "oral": int,
    "esquizoide": int,
    "psicopata": int,
    "masoquista": int,
    "rigido": int
  }}
}}

Retorne apenas o JSON. Não inclua explicações fora dele.
"""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um analista reichiano especialista em linguagem corporal."},
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

        raw = response.choices[0].message.content or ""

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"Falha ao localizar JSON na resposta:\n{raw}")
        cleaned_raw = match.group(0)

        try:
            resultado = json.loads(cleaned_raw)
        except json.JSONDecodeError:
            raise ValueError(f"Falha ao decodificar JSON extraído:\n{cleaned_raw}")

        for parte in PARTES:
            bloco = resultado.get(parte)
            if isinstance(bloco, dict):
                soma = sum(bloco.get(traco, 0) for traco in TRAÇOS)
                if soma != 10:
                    raise ValueError(f"A soma dos traços em '{parte}' é {soma}, mas deveria ser 10.")
                if all(bloco.get(traco, 0) == 0 for traco in TRAÇOS):
                    raise ValueError(f"A parte '{parte}' não possui distribuição significativa entre os traços.")

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
            {"role": "system", "content": "Você é a assistente Lia, uma IA sensível e acolhedora."},
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
        raise HTTPException(status_code=500, detail=f"Falha ao decodificar JSON extraído:\n{cleaned_raw}")

    return {"relatorio": resultado_json}


@app.get("/caracteristicas")
def obter_caracteristicas():
    return {"caracteristicas": CARACTERISTICAS_TEXTO}
