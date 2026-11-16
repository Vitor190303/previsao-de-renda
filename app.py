import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Previsão de renda",
    page_icon="📊",
    layout="centered"
)


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def formatar_real(valor):
    valor_formatado = (
        f"{valor:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return f"R$ {valor_formatado}"


# =========================================================
# CAMINHOS
# =========================================================

pasta_app = (
    Path(__file__)
    .resolve()
    .parent
)

arquivo_dados = (
    pasta_app
    / "input"
    / "previsao_de_renda.csv"
)

pasta_modelo = (
    pasta_app
    / "output"
    / "Modelo"
)


# =========================================================
# CARREGAMENTO DOS DADOS E DO MODELO
# =========================================================

df = pd.read_csv(
    arquivo_dados
)

modelo = joblib.load(
    pasta_modelo
    / "random_forest.pkl"
)

# Aceita os dois nomes que apareceram durante o projeto.
arquivo_colunas = (
    pasta_modelo
    / "columns_modelo.pkl"
)

if not arquivo_colunas.exists():
    arquivo_colunas = (
        pasta_modelo
        / "colunas_modelo.pkl"
    )

colunas_modelo = joblib.load(
    arquivo_colunas
)

categorias = joblib.load(
    pasta_modelo
    / "categorias.pkl"
)


# =========================================================
# CABEÇALHO
# =========================================================

st.title(
    "Previsão de renda"
)

st.write(
    """
    Projeto de Ciência de Dados para análise e previsão
    da renda de clientes utilizando Random Forest.
    """
)


# =========================================================
# ABAS
# =========================================================

aba_analise, aba_previsao = st.tabs(
    [
        "📊 Análises",
        "🤖 Previsão de renda"
    ]
)


# =========================================================
# ABA 1 - ANÁLISES
# =========================================================

with aba_analise:

    st.header(
        "Análise dos dados"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Clientes",
        len(df)
    )

    col2.metric(
        "Renda média",
        formatar_real(
            df["renda"].mean()
        )
    )

    col3.metric(
        "Renda mediana",
        formatar_real(
            df["renda"].median()
        )
    )

    st.divider()

    # -----------------------------------------------------
    # BOXPLOT
    # -----------------------------------------------------

    st.subheader(
        "Renda por sexo"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=df,
        x="sexo",
        y="renda",
        ax=ax
    )

    ax.set_yscale(
        "log"
    )

    ax.set_title(
        "Distribuição da renda por sexo"
    )

    ax.set_xlabel(
        "Sexo"
    )

    ax.set_ylabel(
        "Renda"
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )

    st.divider()

    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    st.subheader(
        "Correlação entre variáveis numéricas"
    )

    variaveis_numericas = [
        "idade",
        "tempo_emprego",
        "qtd_filhos",
        "qt_pessoas_residencia",
        "renda"
    ]

    correlacao = (
        df[
            variaveis_numericas
        ]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.heatmap(
        correlacao,
        annot=True,
        fmt=".2f",
        ax=ax
    )

    ax.set_title(
        "Correlação entre variáveis numéricas"
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )

    st.divider()

    # -----------------------------------------------------
    # MÉTRICAS DO MODELO
    # -----------------------------------------------------

    st.subheader(
        "Desempenho do modelo"
    )

    metrica1, metrica2, metrica3 = st.columns(3)

    metrica1.metric(
        "MAE",
        "R$ 2.868,79"
    )

    metrica2.metric(
        "RMSE",
        "R$ 5.625,31"
    )

    metrica3.metric(
        "R²",
        "0,354"
    )

    st.caption(
        "Resultados obtidos pela Random Forest na base de teste."
    )

    st.divider()

    # -----------------------------------------------------
    # IMPORTÂNCIA DAS VARIÁVEIS
    # -----------------------------------------------------

    st.subheader(
        "Variáveis mais importantes para o modelo"
    )

    importancias = pd.Series(
        modelo.feature_importances_,
        index=colunas_modelo
    ).sort_values(
        ascending=False
    )

    top_importancias = (
        importancias
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        top_importancias.index,
        top_importancias.values
    )

    ax.set_xlabel(
        "Importância"
    )

    ax.set_ylabel(
        "Variável"
    )

    ax.set_title(
        "Top 10 variáveis mais importantes"
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )

    st.info(
        """
        As variáveis mais relevantes para a Random Forest
        foram principalmente tempo de emprego e sexo.

        A importância representa o quanto cada variável foi
        utilizada pelo modelo durante suas decisões e não
        deve ser interpretada como relação de causa e efeito.
        """
    )


# =========================================================
# ABA 2 - PREVISÃO
# =========================================================

with aba_previsao:

    st.header(
        "Previsão de renda"
    )

    st.info(
        """
        Informe as características do cliente para obter uma
        estimativa de renda.

        O valor apresentado é uma previsão do modelo e não
        representa necessariamente a renda exata do cliente.
        """
    )

    with st.form(
        "formulario_cliente"
    ):

        sexo = st.selectbox(
            "Sexo",
            categorias["sexo"]
        )

        idade = st.number_input(
            "Idade",
            min_value=18,
            max_value=100,
            value=35
        )

        posse_veiculo = st.checkbox(
            "Possui veículo"
        )

        posse_imovel = st.checkbox(
            "Possui imóvel"
        )

        qtd_filhos = st.number_input(
            "Quantidade de filhos",
            min_value=0,
            max_value=20,
            value=0
        )

        tipo_renda = st.selectbox(
            "Tipo de renda",
            categorias["tipo_renda"]
        )

        educacao = st.selectbox(
            "Educação",
            categorias["educacao"]
        )

        estado_civil = st.selectbox(
            "Estado civil",
            categorias["estado_civil"]
        )

        tipo_residencia = st.selectbox(
            "Tipo de residência",
            categorias["tipo_residencia"]
        )

        tempo_emprego = st.number_input(
            "Tempo de emprego (anos)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5
        )

        qt_pessoas_residencia = st.number_input(
            "Quantidade de pessoas na residência",
            min_value=1,
            max_value=20,
            value=2
        )

        enviar = st.form_submit_button(
            "Prever renda"
        )

    if enviar:

        cliente = pd.DataFrame({
            "sexo": [sexo],
            "posse_de_veiculo": [posse_veiculo],
            "posse_de_imovel": [posse_imovel],
            "qtd_filhos": [qtd_filhos],
            "tipo_renda": [tipo_renda],
            "educacao": [educacao],
            "estado_civil": [estado_civil],
            "tipo_residencia": [tipo_residencia],
            "idade": [idade],
            "tempo_emprego": [tempo_emprego],
            "qt_pessoas_residencia": [
                qt_pessoas_residencia
            ]
        })

        cliente_modelo = pd.get_dummies(
            cliente,
            dtype=int
        )

        cliente_modelo = cliente_modelo.reindex(
            columns=colunas_modelo,
            fill_value=0
        )

        previsao_log = modelo.predict(
            cliente_modelo
        )[0]

        previsao_renda = np.exp(
            previsao_log
        )

        st.success(
            f"Renda estimada: {formatar_real(previsao_renda)}"
        )

        st.caption(
            "Modelo utilizado: Random Forest Regressor"
        )

        st.caption(
            "R² do modelo na base de teste: aproximadamente 0,354"
        )
