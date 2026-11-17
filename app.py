"""Aplicativo educacional: o mesmo pipeline atende treino e previsão."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from modelagem import VARIAVEIS, importancias, prever, treinar_e_avaliar

st.set_page_config(page_title="Previsão de renda", page_icon="📊", layout="wide")

def reais(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

@st.cache_resource(show_spinner="Treinando modelo e validando por cliente...")
def iniciar():
    base = pd.read_csv(Path(__file__).parent / "input" / "previsao_de_renda.csv")
    modelo, metricas, diagnosticos = treinar_e_avaliar(base)
    return base, modelo, metricas, diagnosticos

st.title("📊 Previsão de renda")
st.caption("Estudo de Machine Learning • validação por clientes não vistos")
st.warning("Estimativas experimentais: não utilizar para decisões de crédito, contratação ou elegibilidade.")
try:
    base, modelo, metricas, diagnosticos = iniciar()
except (OSError, ValueError) as erro:
    st.error(f"Erro ao iniciar o projeto: {erro}")
    st.stop()

aba_analise, aba_previsao = st.tabs(["📈 Análises", "🤖 Previsão"])
with aba_analise:
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(base))
    col2.metric("Clientes únicos", base["id_cliente"].nunique())
    col3.metric("Clientes compartilhados entre treino e teste", metricas["clientes_em_ambos"])
    a, b, c = st.columns(3)
    a.metric("MAE", reais(metricas["mae"]))
    b.metric("RMSE", reais(metricas["rmse"]))
    c.metric("R²", f"{metricas['r2']:.4f}")
    st.caption(f"Treino: {metricas['registros_treino']} registros / {metricas['clientes_treino']} clientes. "
               f"Teste: {metricas['registros_teste']} registros / {metricas['clientes_teste']} clientes. "
               "Métricas calculadas no teste, em reais; não são valores fixos.")
    st.info("R² não é acurácia. Importância de variáveis não significa causalidade.")
    st.subheader("Variáveis mais utilizadas pela Random Forest")
    valores = importancias(modelo).head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(valores.index, valores.values)
    ax.set_xlabel("Importância por redução de impureza (não causal)")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Erros por faixa de renda real")
    analise = diagnosticos.copy()
    analise["faixa"] = pd.cut(analise["renda"],
        bins=[0, 2000, 5000, 10000, 20000, np.inf],
        labels=["Até R$ 2 mil", "R$ 2–5 mil", "R$ 5–10 mil", "R$ 10–20 mil", "Acima R$ 20 mil"])
    resumo = analise.groupby("faixa", observed=True).agg(
        registros=("renda", "count"), erro_medio=("erro_absoluto", "mean"),
        vies_medio=("erro_assinado", "mean"),
        renda_real=("renda", "mean"), renda_prevista=("previsto", "mean"))
    st.dataframe(resumo.round(2), use_container_width=True)
    st.caption("Faixas usam renda REAL do teste, não conhecida para clientes novos. Quantidades são registros, não necessariamente clientes.")
with aba_previsao:
    st.subheader("Estimar a renda de um cliente")
    st.caption("O mesmo pipeline de imputação, codificação e regressão é utilizado na avaliação e na previsão.")
    categorias = ["sexo", "tipo_renda", "educacao", "estado_civil", "tipo_residencia"]
    opcoes = {c: sorted(base[c].dropna().astype(str).unique()) for c in categorias}
    with st.form("cliente"):
        c1, c2 = st.columns(2)
        with c1:
            sexo = st.selectbox("Sexo", opcoes["sexo"])
            idade = st.number_input("Idade", 18, 100, 35)
            tipo_renda = st.selectbox("Tipo de renda", opcoes["tipo_renda"])
            educacao = st.selectbox("Escolaridade", opcoes["educacao"])
            estado_civil = st.selectbox("Estado civil", opcoes["estado_civil"])
            tipo_residencia = st.selectbox("Tipo de residência", opcoes["tipo_residencia"])
        with c2:
            veiculo = st.checkbox("Possui veículo")
            imovel = st.checkbox("Possui imóvel")
            filhos = st.number_input("Quantidade de filhos", 0, 20, 0)
            nao_informado = st.checkbox("Tempo de emprego não informado")
            emprego = st.number_input("Tempo de emprego (anos)", 0.0, 60.0, 5.0, 0.5, disabled=nao_informado)
            pessoas = st.number_input("Quantidade de pessoas na residência", 1, 20, 2)
        enviar = st.form_submit_button("Prever renda")
    if enviar:
        cliente = pd.DataFrame([{
            "sexo": sexo, "idade": idade, "tipo_renda": tipo_renda,
            "educacao": educacao, "estado_civil": estado_civil,
            "tipo_residencia": tipo_residencia, "posse_de_veiculo": veiculo,
            "posse_de_imovel": imovel, "qtd_filhos": filhos,
            "tempo_emprego": np.nan if nao_informado else emprego,
            "qt_pessoas_residencia": pessoas
        }])
        st.success("Renda estimada: " + reais(float(prever(modelo, cliente)[0])))
        st.caption("O resultado é uma estimativa sujeita a erros relevantes.")
