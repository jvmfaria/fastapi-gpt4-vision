from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt

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

# Lê os textos dos traços de caráter
def carregar_tracos():
    tracos = {}
    nomes = ["esquizoide", "masoquista", "oral", "psicopata", "rigido"]
    for nome in nomes:
        try:
            with open(f"tracos/{nome}.txt", "r", encoding="utf-8") as f:
                tracos[nome] = f.read()
        except FileNotFoundError:
            tracos[nome] = f"(Descrição do traço {nome} não encontrada.)"
    return tracos

# Formata a resposta como mensagem de WhatsApp
def formatar_resposta(result):
    return (
        "🧠 *Análise dos Traços de Caráter*\n\n"
        f"🔹 *Oral*: {result['oral']}\n"
        f"🔹 *Esquizoide*: {result['esquizoide']}\n"
        f"🔹 *Masoquista*: {result['masoquista']}\n"
        f"🔹 *Psicopata*: {result['psicopata']}\n"
        f"🔹 *Rígido*: {result['rigido']}\n\n"
        f"📝 *Explicação:*\n{result['explicacao']}"
    )

# Gera gráfico de colunas e retorna como base64
def gerar_grafico_base64(result):
    labels = ["Oral", "Esquizoide", "Masoquista", "Psicopata", "Rígido"]
    valores = [
        result["oral"],
        result["esquizoide"],
        result["masoquista"],
        result["psicopata"],
        result["rigido"]
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, valores, color="#4A90E2")
    plt.ylim(0, 10)
    plt.title("Traços de Caráter – Análise Facial")
    plt.xlabel("Traços")
    plt.ylabel("Pontuação")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, int(yval), ha="center", fontsize=10)

    plt.tight_layout()
    caminho = "grafico.png"
    plt.savefig(caminho)
    plt.close()

    with open(caminho, "rb") as img_file:
        grafico_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return grafico_base64

# Endpoint principal
@app.post("/classificar")
async def classificar(imagem: UploadFile = File(...)):
    try:
        image_data_url = file_to_data_url(imagem)
        tracos = carregar_tracos()

        tracos_texto = "\n".join(
            f"{nome.capitalize()}:\n{descricao}" for nome, descricao in tracos.items()
        )

        prompt = (
            "Você é um analista reichiano experiente e também domina profundamente os conceitos do estudo 'O Corpo Explica'. "
            "A seguir estão os resumos dos cinco tipos de caráter segundo esses estudos:\n\n"
            f"{tracos_texto}\n\n"
            "Com base na imagem facial fornecida, analise e classifique os traços da pessoa nos tipos de caráter: "
            "oral, esquizóide, masoquista, psicopata e rígido. Para cada tipo, atribua uma pontuação de 0 a 10, sendo que a soma total deve ser exatamente 10.\n"
            "Retorne os resultados no formato JSON abaixo, seguido de uma breve explicação baseada nos estudos fornecidos:\n"
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
            cleaned_raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
            resultado = json.loads(cleaned_raw)

            mensagem = formatar_resposta(resultado)
            grafico_base64 = gerar_grafico_base64(resultado)

            return {
                "mensagem": mensagem,
                "grafico_base64": grafico_base64  # imagem para enviar pelo WhatsApp como mídia
            }

        except json.JSONDecodeError:
            return {
                "erro": "A resposta não está em formato JSON válido.",
                "resposta_bruta": raw
            }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
