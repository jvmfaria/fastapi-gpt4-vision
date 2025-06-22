from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis do ambiente
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

BASE_DIR = os.getenv("BASE_DIR", "./app/caracteristicas")
FOTOS_BASE_URL = "https://raw.githubusercontent.com/jvmfaria/fastapi-gpt4-vision/main/app/fotos"

TRAÇOS = ["esquizoide", "masoquista", "oral", "psicopata", "rigido"]
PARTES = ["cabeca", "olhos", "boca", "tronco", "quadril", "pernas"]

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
            for traco in TRAÇOS:
                ponto = bloco.get(traco, 0)
                explicacao = bloco.get("explicacao", {})
                justificativa = explicacao.get(traco, "")
                mensagem.append(f"• {traco.capitalize()}: {ponto} — {justificativa}")
    mensagem.append("\n🧠 *Total por Traço*")
    for traco in TRAÇOS:
        total = dados.get("soma_total_por_traco", {}).get(traco, 0)
        mensagem.append(f"• {traco.capitalize()}: {total}")
    mensagem.append("\n📌 *Metodologia*: lia.ai!")
    return "\n".join(mensagem)

def distribuicoes_iguais(dados):
    distros = set()
    for parte in PARTES:
        bloco = dados.get(parte, {})
        dist = tuple(bloco.get(traco, 0) for traco in TRAÇOS)
        if dist in distros:
            return True
        distros.add(dist)
    return False

def normalizar_justificativas(dados):
    for parte in PARTES:
        bloco = dados.get(parte, {})
        explicacoes = bloco.get("explicacao", {})
        for traco, texto in explicacoes.items():
            texto = texto.strip().capitalize()
            if not texto.endswith("."):
                texto += "."
            explicacoes[traco] = texto
    return dados

def comparar_com_historico(dados_atuais, historico):
    comparacao = {}
    for traco in TRAÇOS:
        atual = dados_atuais.get("soma_total_por_traco", {}).get(traco, 0)
        anterior = historico.get("soma_total_por_traco", {}).get(traco, 0)
        comparacao[traco] = {
            "anterior": anterior,
            "atual": atual,
            "diferenca": atual - anterior
        }
    return comparacao

def construir_dados_classificacao(resultado: dict):
    totais = resultado.get("soma_total_por_traco", {})
    tracos_ordenados = sorted(totais.items(), key=lambda x: x[1], reverse=True)
    tracos_dominantes = [traco for traco, _ in tracos_ordenados[:3]]

    explicacoes = []
    for parte in PARTES:
        explicacao = resultado.get(parte, {}).get("explicacao", {})
        for traco in tracos_dominantes:
            if traco in explicacao:
                explicacoes.append(explicacao[traco])

    return {
        "tracos": tracos_dominantes,
        "dores": ["carência emocional", "necessidade de controle", "medo de rejeição"],
        "recursos": ["capacidade de se adaptar", "empatia", "autopercepção corporal"],
        "padroes_dependencia": ["busca constante por aprovação", "medo de romper vínculos afetivos"],
        "escolhas_inconscientes": [{
            "decisao": "reprimir vontades para manter vínculos",
            "origem": "experiências infantis de rejeição emocional"
        }],
        "impactos": ["tensões no tórax e mandíbula", "bloqueios na expressão emocional"]
    }

def gerar_prompt_relatorio(dados_classificacao, nome_cliente, data_atendimento, genero_cliente):
    pronome = "o" if genero_cliente.lower() == "masculino" else "a"
    artigo = "do" if genero_cliente.lower() == "masculino" else "da"

    tracos = dados_classificacao.get("tracos", [])
    dores = dados_classificacao.get("dores", [])
    recursos = dados_classificacao.get("recursos", [])
    padroes_dependencia = dados_classificacao.get("padroes_dependencia", [])
    escolhas_inconscientes = dados_classificacao.get("escolhas_inconscientes", [])
    impactos = dados_classificacao.get("impactos", [])

    return f"""
Você é a assistente Lia – Linguagem Integrativa de Autoconhecimento, da Corphus.

Sua tarefa é gerar um relatório completo, humanizado e terapêutico de análise corporal no formato JSON, com base na psicologia reichiana, bioenergética e leitura corporal.

⚠️ O resultado deve ser um objeto JSON estruturado com os seguintes campos, todos preenchidos com profundidade, empatia e linguagem acolhedora. 

Considere o histórico emocional, padrões inconscientes e a estrutura corporal de {nome_cliente}, respeitando sua trajetória única. Adapte a linguagem de acordo com o gênero: utilize formas no feminino se for mulher e no masculino se for homem.

Responda apenas com o objeto JSON, conforme o modelo abaixo:

```json
{{
  "cabecalho": {{
    "nome_cliente": "{nome_cliente}",
    "data_atendimento": "{data_atendimento}",
    "nome_analista": "Márcio Conceição",
    "titulo": "Relatório de Análise da Sua História"
  }},
  "objetivo": "Descreva o propósito central deste processo terapêutico para {pronome} cliente...",
  "resumo_inicial": "{nome_cliente}, a sua história revela...",
  "dores_e_recursos": {{
    "dores": {json.dumps(dores)},
    "recursos": {json.dumps(recursos)}
  }},
  "tracos_que_explicam": "Os principais traços que influenciam {artigo} cliente são: {', '.join(tracos)}...",
  "padroes_dependencia_emocional": {json.dumps(padroes_dependencia)},
  "escolhas_inconscientes": [
    {{
      "decisao": "{escolhas_inconscientes[0]['decisao'] if escolhas_inconscientes else ''}",
      "origem": "{escolhas_inconscientes[0]['origem'] if escolhas_inconscientes else ''}"
    }}
  ],
  "impactos_das_dores": {json.dumps(impactos)},
  "virada_de_chave": "Apresente o ponto de virada mais significativo...",
  "proximos_passos": ["Sugira caminhos terapêuticos..."],
  "acoes_praticas": [
    {{
      "oque": "Indique uma ação concreta...",
      "como": "Explique como ela pode ser feita...",
      "porque": "Justifique o impacto positivo..."
    }}
  ],
  "conclusao": "Finalize com uma mensagem que reconheça..."
}}
```"""

@app.post("/classificar")
async def classificar():
    try:
        frente_url = f"{FOTOS_BASE_URL}/imagem_frente.png"
        lado_url = f"{FOTOS_BASE_URL}/imagem_lado.png"
        costas_url = f"{FOTOS_BASE_URL}/imagem_costas.png"

        prompt_instrucoes = """
Você é um analista reichiano altamente experiente.

Abaixo estão as descrições referenciais completas de cada traço de caráter, detalhadas por parte do corpo:

<<CARACTERISTICAS>>

Sua tarefa é analisar cuidadosamente as imagens corporais fornecidas (frente, lado e costas)...

Apenas o JSON. Nada mais.
""".replace("<<CARACTERISTICAS>>", CARACTERISTICAS_TEXTO)

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um analista reichiano altamente especialista em linguagem corporal."},
                {"role": "user", "content": prompt_instrucoes},
                {"role": "user", "content": [
                    {"type": "text", "text": "Imagem de frente:"},
                    {"type": "image_url", "image_url": {"url": frente_url}},
                    {"type": "text", "text": "Imagem de lateral:"},
                    {"type": "image_url", "image_url": {"url": lado_url}},
                    {"type": "text", "text": "Imagem de costas:"},
                    {"type": "image_url", "image_url": {"url": costas_url}}
                ]}
            ],
            temperature=0,
            max_tokens=2500
        )

        raw = response.choices[0].message.content or ""
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"Falha ao localizar JSON na resposta:\n{raw}")
        resultado = json.loads(match.group(0))

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
    resultado_classificacao = payload.get("resultado_classificacao")
    genero_cliente = payload.get("genero_cliente")
    data_atendimento = payload.get("data_atendimento", datetime.today().strftime("%d/%m/%Y"))

    if not resultado_classificacao:
        raise HTTPException(status_code=400, detail="Resultado de classificação ausente.")
    if not genero_cliente:
        raise HTTPException(status_code=400, detail="Gênero do cliente ausente.")

    dados_classificacao = construir_dados_classificacao(resultado_classificacao)
    prompt = gerar_prompt_relatorio(dados_classificacao, nome_cliente, data_atendimento, genero_cliente)

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
