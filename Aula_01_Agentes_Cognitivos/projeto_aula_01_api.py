"""
🏦 Projeto Aula 01 - API do Analista Financeiro
================================================

API REST com FastAPI para interagir com o Agente Analista Financeiro.

Funcionalidades:
- Chat com memória de sessão
- Consulta de cotações de ações
- Análise de crédito de clientes
- Cálculo de risco de operações
- Conversão de moedas

Para executar:
    uvicorn projeto_aula_01_api:app --reload

Documentação interativa:
    http://localhost:8000/docs
"""

import os
import random
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuração da API Key (defina antes de importar langchain)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "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"  # Substitua pela sua chave

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import warnings

# Suprime avisos de deprecação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# FERRAMENTAS DO AGENTE
# =============================================================================

class CotacaoInput(BaseModel):
    """Input para consulta de cotação de ações."""
    simbolo: str = Field(description="Símbolo da ação (ex: PETR4, VALE3, ITUB4)")

@tool(args_schema=CotacaoInput)
def consultar_cotacao(simbolo: str) -> dict:
    """
    Consulta a cotação atual de uma ação na B3.
    Use esta ferramenta quando precisar saber o preço atual de uma ação brasileira.
    """
    acoes_simuladas = {
        "PETR4": {"nome": "Petrobras PN", "preco_base": 38.50, "variacao_max": 2.0},
        "VALE3": {"nome": "Vale ON", "preco_base": 62.30, "variacao_max": 3.0},
        "ITUB4": {"nome": "Itaú Unibanco PN", "preco_base": 32.80, "variacao_max": 1.5},
        "BBDC4": {"nome": "Bradesco PN", "preco_base": 12.45, "variacao_max": 0.8},
        "ABEV3": {"nome": "Ambev ON", "preco_base": 11.20, "variacao_max": 0.5},
        "WEGE3": {"nome": "WEG ON", "preco_base": 52.60, "variacao_max": 2.5},
        "MGLU3": {"nome": "Magazine Luiza ON", "preco_base": 2.15, "variacao_max": 0.3},
        "B3SA3": {"nome": "B3 ON", "preco_base": 10.85, "variacao_max": 0.6},
    }
    
    simbolo_upper = simbolo.upper()
    
    if simbolo_upper not in acoes_simuladas:
        return {
            "erro": f"Ação '{simbolo}' não encontrada.",
            "acoes_disponiveis": list(acoes_simuladas.keys())
        }
    
    dados = acoes_simuladas[simbolo_upper]
    variacao = random.uniform(-dados["variacao_max"], dados["variacao_max"])
    preco_atual = round(dados["preco_base"] + variacao, 2)
    variacao_percentual = round((variacao / dados["preco_base"]) * 100, 2)
    
    return {
        "simbolo": simbolo_upper,
        "nome": dados["nome"],
        "preco_atual": preco_atual,
        "moeda": "BRL",
        "variacao_dia": variacao_percentual,
        "horario_consulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "mercado_aberto" if 10 <= datetime.now().hour < 17 else "mercado_fechado"
    }


class ClienteInput(BaseModel):
    """Input para consulta de dados do cliente."""
    cpf: str = Field(description="CPF do cliente (apenas números)")

@tool(args_schema=ClienteInput)
def consultar_cliente(cpf: str) -> dict:
    """
    Consulta os dados financeiros de um cliente no sistema interno.
    Use esta ferramenta para obter informações sobre score de crédito, renda e histórico.
    """
    clientes_simulados = {
        "12345678900": {
            "nome": "Maria Silva",
            "score_credito": 820,
            "renda_mensal": 15000.00,
            "comprometimento_renda": 0.25,
            "tempo_conta_anos": 8,
            "historico_atrasos": 0,
            "perfil_investidor": "Moderado"
        },
        "98765432100": {
            "nome": "João Santos",
            "score_credito": 650,
            "renda_mensal": 5500.00,
            "comprometimento_renda": 0.45,
            "tempo_conta_anos": 2,
            "historico_atrasos": 3,
            "perfil_investidor": "Conservador"
        },
        "11122233344": {
            "nome": "Ana Oliveira",
            "score_credito": 750,
            "renda_mensal": 25000.00,
            "comprometimento_renda": 0.15,
            "tempo_conta_anos": 12,
            "historico_atrasos": 1,
            "perfil_investidor": "Arrojado"
        },
    }
    
    cpf_limpo = cpf.replace(".", "").replace("-", "").replace(" ", "")
    
    if cpf_limpo not in clientes_simulados:
        return {
            "erro": f"Cliente com CPF '{cpf}' não encontrado no sistema.",
            "sugestao": "Verifique se o CPF está correto ou se o cliente está cadastrado."
        }
    
    dados = clientes_simulados[cpf_limpo].copy()
    dados["cpf"] = cpf_limpo
    dados["data_consulta"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return dados


class RiscoInput(BaseModel):
    """Input para cálculo de risco de operação."""
    valor_operacao: float = Field(description="Valor da operação em reais")
    prazo_meses: int = Field(description="Prazo da operação em meses")
    score_cliente: int = Field(description="Score de crédito do cliente (0-1000)")
    comprometimento_atual: float = Field(description="Percentual atual de comprometimento da renda (0.0 a 1.0)")

@tool(args_schema=RiscoInput)
def calcular_risco(valor_operacao: float, prazo_meses: int, score_cliente: int, comprometimento_atual: float) -> dict:
    """
    Calcula o risco de uma operação de crédito baseado em múltiplos fatores.
    Retorna uma classificação de risco e recomendação.
    """
    risco_score = 0
    fatores = []
    
    if score_cliente >= 800:
        risco_score += 5
        fatores.append("Score excelente: +5 pontos de risco")
    elif score_cliente >= 700:
        risco_score += 15
        fatores.append("Score bom: +15 pontos de risco")
    elif score_cliente >= 600:
        risco_score += 35
        fatores.append("Score médio: +35 pontos de risco")
    else:
        risco_score += 50
        fatores.append("Score baixo: +50 pontos de risco")
    
    if comprometimento_atual <= 0.3:
        risco_score += 5
        fatores.append("Comprometimento baixo (<=30%): +5 pontos de risco")
    elif comprometimento_atual <= 0.5:
        risco_score += 20
        fatores.append("Comprometimento médio (30-50%): +20 pontos de risco")
    else:
        risco_score += 40
        fatores.append("Comprometimento alto (>50%): +40 pontos de risco")
    
    if prazo_meses <= 12:
        risco_score += 5
        fatores.append("Prazo curto (<=12 meses): +5 pontos de risco")
    elif prazo_meses <= 36:
        risco_score += 10
        fatores.append("Prazo médio (12-36 meses): +10 pontos de risco")
    else:
        risco_score += 20
        fatores.append("Prazo longo (>36 meses): +20 pontos de risco")
    
    if risco_score <= 25:
        classificacao = "BAIXO"
        recomendacao = "Aprovar - Operação de baixo risco"
        taxa_sugerida = 1.2
    elif risco_score <= 50:
        classificacao = "MÉDIO"
        recomendacao = "Aprovar com cautela - Considerar garantias adicionais"
        taxa_sugerida = 1.8
    elif risco_score <= 75:
        classificacao = "ALTO"
        recomendacao = "Revisar - Necessita aprovação de comitê"
        taxa_sugerida = 2.5
    else:
        classificacao = "MUITO ALTO"
        recomendacao = "Não aprovar - Risco excessivo"
        taxa_sugerida = None
    
    return {
        "score_risco": risco_score,
        "classificacao": classificacao,
        "recomendacao": recomendacao,
        "taxa_juros_mensal_sugerida": taxa_sugerida,
        "valor_operacao": valor_operacao,
        "prazo_meses": prazo_meses,
        "fatores_analisados": fatores
    }


class ConversaoInput(BaseModel):
    """Input para conversão de moedas."""
    valor: float = Field(description="Valor a ser convertido")
    moeda_origem: str = Field(description="Código da moeda de origem (USD, EUR, BRL)")
    moeda_destino: str = Field(description="Código da moeda de destino (USD, EUR, BRL)")

@tool(args_schema=ConversaoInput)
def converter_moeda(valor: float, moeda_origem: str, moeda_destino: str) -> dict:
    """
    Converte valores entre moedas usando cotações atualizadas.
    Moedas suportadas: USD (Dólar), EUR (Euro), BRL (Real).
    """
    cotacoes_para_brl = {
        "USD": 5.45 + random.uniform(-0.1, 0.1),
        "EUR": 5.95 + random.uniform(-0.1, 0.1),
        "BRL": 1.0,
    }
    
    moeda_origem = moeda_origem.upper()
    moeda_destino = moeda_destino.upper()
    
    if moeda_origem not in cotacoes_para_brl or moeda_destino not in cotacoes_para_brl:
        return {
            "erro": "Moeda não suportada",
            "moedas_disponiveis": list(cotacoes_para_brl.keys())
        }
    
    valor_em_brl = valor * cotacoes_para_brl[moeda_origem]
    valor_convertido = valor_em_brl / cotacoes_para_brl[moeda_destino]
    taxa_conversao = cotacoes_para_brl[moeda_origem] / cotacoes_para_brl[moeda_destino]
    
    return {
        "valor_original": valor,
        "moeda_origem": moeda_origem,
        "valor_convertido": round(valor_convertido, 2),
        "moeda_destino": moeda_destino,
        "taxa_conversao": round(taxa_conversao, 4),
        "cotacao_dolar_brl": round(cotacoes_para_brl["USD"], 4),
        "cotacao_euro_brl": round(cotacoes_para_brl["EUR"], 4),
        "horario_cotacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# =============================================================================
# CONFIGURAÇÃO DO AGENTE
# =============================================================================

SYSTEM_PROMPT = """Você é um Analista Financeiro Sênior, especializado em análise de mercado e crédito do setor bancário brasileiro.

## Suas Competências:
1. **Análise de Mercado**: Consultar e interpretar cotações de ações da B3
2. **Análise de Crédito**: Avaliar perfil de clientes e calcular riscos de operações
3. **Câmbio**: Realizar conversões entre moedas com cotações atualizadas
4. **Consultoria**: Fornecer recomendações embasadas em dados

## Diretrizes Operacionais:
- SEMPRE utilize as ferramentas disponíveis para obter dados atualizados
- NUNCA invente dados financeiros - use apenas informações verificadas via ferramentas
- Seja PRECISO e PROFISSIONAL em todas as análises
- JUSTIFIQUE recomendações com dados concretos obtidos
- Se houver INCERTEZA, comunique claramente as limitações

## Padrão de Resposta:
- Estruture respostas de forma clara e organizada
- Use formatação (bullets, números) quando apropriado
- Inclua sempre os dados que fundamentam sua análise
"""

# Lista de ferramentas
ferramentas = [consultar_cotacao, consultar_cliente, calcular_risco, converter_moeda]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Memory Saver para persistência de estado
memory = MemorySaver()

# Cria o agente
agente = create_react_agent(llm, ferramentas, prompt=SYSTEM_PROMPT, checkpointer=memory)

# Store para histórico de sessões
session_store: Dict[str, List[dict]] = {}


# =============================================================================
# MODELOS PYDANTIC PARA API
# =============================================================================

class ChatRequest(BaseModel):
    """Requisição de chat."""
    message: str = Field(..., description="Mensagem do usuário")
    session_id: str = Field(default="default", description="ID da sessão para memória")

class ChatResponse(BaseModel):
    """Resposta do chat."""
    response: str = Field(..., description="Resposta do agente")
    session_id: str = Field(..., description="ID da sessão")
    tools_used: List[str] = Field(default=[], description="Ferramentas utilizadas")

class ToolInfo(BaseModel):
    """Informações de uma ferramenta."""
    name: str
    description: str

class SessionInfo(BaseModel):
    """Informações de uma sessão."""
    session_id: str
    message_count: int


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="🏦 Analista Financeiro API",
    description="""
API REST para interagir com o Agente Analista Financeiro.

## Funcionalidades

* **Chat com Memória**: Converse com o agente mantendo contexto entre mensagens
* **Consulta de Cotações**: Obtenha cotações de ações da B3
* **Análise de Crédito**: Consulte dados de clientes e calcule riscos
* **Conversão de Moedas**: Converta valores entre USD, EUR e BRL

## Sessões

Use o parâmetro `session_id` para manter contexto entre mensagens.
Cada sessão mantém o histórico da conversa.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Endpoint de boas-vindas."""
    return {
        "message": "🏦 Bem-vindo à API do Analista Financeiro!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat - Conversar com o agente",
            "tools": "GET /tools - Listar ferramentas disponíveis",
            "sessions": "GET /sessions - Listar sessões ativas",
            "clear_session": "DELETE /sessions/{session_id} - Limpar sessão"
        }
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Envia uma mensagem para o agente e recebe uma resposta.
    
    O agente tem acesso a ferramentas de:
    - Consulta de cotações de ações
    - Consulta de dados de clientes
    - Cálculo de risco de operações
    - Conversão de moedas
    
    Use o mesmo `session_id` para manter contexto entre mensagens.
    """
    try:
        config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}
        
        # Executa o agente
        resultado = agente.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        # Extrai ferramentas utilizadas
        tools_used = []
        for msg in resultado["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc['name'])
        
        # Armazena no histórico
        if request.session_id not in session_store:
            session_store[request.session_id] = []
        session_store[request.session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        session_store[request.session_id].append({
            "role": "assistant",
            "content": resultado["messages"][-1].content,
            "tools_used": tools_used,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=resultado["messages"][-1].content,
            session_id=request.session_id,
            tools_used=tools_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar mensagem: {str(e)}")


@app.get("/tools", response_model=List[ToolInfo], tags=["Tools"])
async def list_tools():
    """Lista todas as ferramentas disponíveis para o agente."""
    return [
        ToolInfo(name=tool.name, description=tool.description)
        for tool in ferramentas
    ]


@app.get("/sessions", response_model=List[SessionInfo], tags=["Sessions"])
async def list_sessions():
    """Lista todas as sessões ativas com contagem de mensagens."""
    return [
        SessionInfo(session_id=sid, message_count=len(msgs))
        for sid, msgs in session_store.items()
    ]


@app.get("/sessions/{session_id}/history", tags=["Sessions"])
async def get_session_history(session_id: str):
    """Retorna o histórico completo de uma sessão."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail=f"Sessão '{session_id}' não encontrada")
    
    return {
        "session_id": session_id,
        "messages": session_store[session_id]
    }


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def clear_session(session_id: str):
    """Limpa o histórico de uma sessão específica."""
    if session_id in session_store:
        del session_store[session_id]
        return {"message": f"Sessão '{session_id}' limpa com sucesso"}
    
    raise HTTPException(status_code=404, detail=f"Sessão '{session_id}' não encontrada")


# =============================================================================
# ENDPOINTS DIRETOS DAS FERRAMENTAS (para testes)
# =============================================================================

@app.get("/cotacao/{simbolo}", tags=["Tools - Direto"])
async def get_cotacao(simbolo: str):
    """Consulta direta de cotação de uma ação."""
    return consultar_cotacao.invoke({"simbolo": simbolo})


@app.get("/cliente/{cpf}", tags=["Tools - Direto"])
async def get_cliente(cpf: str):
    """Consulta direta de dados de um cliente."""
    return consultar_cliente.invoke({"cpf": cpf})


@app.post("/risco", tags=["Tools - Direto"])
async def post_risco(
    valor_operacao: float,
    prazo_meses: int,
    score_cliente: int,
    comprometimento_atual: float
):
    """Cálculo direto de risco de operação."""
    return calcular_risco.invoke({
        "valor_operacao": valor_operacao,
        "prazo_meses": prazo_meses,
        "score_cliente": score_cliente,
        "comprometimento_atual": comprometimento_atual
    })


@app.get("/conversao", tags=["Tools - Direto"])
async def get_conversao(valor: float, moeda_origem: str, moeda_destino: str):
    """Conversão direta de moedas."""
    return converter_moeda.invoke({
        "valor": valor,
        "moeda_origem": moeda_origem,
        "moeda_destino": moeda_destino
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🏦 ANALISTA FINANCEIRO API")
    print("=" * 60)
    print("\n📍 Iniciando servidor...")
    print("📖 Documentação: http://localhost:8000/docs")
    print("🔄 Swagger UI: http://localhost:8000/redoc")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
