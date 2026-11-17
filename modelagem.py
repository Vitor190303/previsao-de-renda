"""Fluxo único de treino, validação e previsão (dados por cliente)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ALVO = "renda"
CATEGORICAS = ["sexo", "tipo_renda", "educacao", "estado_civil", "tipo_residencia"]
NUMERICAS = ["posse_de_veiculo", "posse_de_imovel", "qtd_filhos", "idade", "tempo_emprego", "qt_pessoas_residencia"]
VARIAVEIS = CATEGORICAS + NUMERICAS

def validar(base):
    faltantes = set(VARIAVEIS + ["renda", "id_cliente"]) - set(base.columns)
    if faltantes:
        raise ValueError(f"Colunas faltantes: {sorted(faltantes)}")
    dados = base.copy()
    dados["renda"] = pd.to_numeric(dados["renda"], errors="coerce")
    dados = dados.dropna(subset=["renda", "id_cliente"]).copy()
    dados = dados.loc[np.isfinite(dados["renda"]) & (dados["renda"] > 0)].copy()
    for coluna in NUMERICAS:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    if dados["id_cliente"].nunique() < 5:
        raise ValueError("Dados insuficientes para separação por cliente.")
    return dados

def criar_modelo():
    tratamento = ColumnTransformer([
        ("categoricas", Pipeline([
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAS),
        ("numericas", SimpleImputer(strategy="median"), NUMERICAS)
    ], remainder="drop")
    floresta = RandomForestRegressor(n_estimators=300, max_depth=8,
                                   min_samples_leaf=10, random_state=42, n_jobs=-1)
    regressor = Pipeline([("tratamento", tratamento), ("floresta", floresta)])
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)

def treinar_e_avaliar(base):
    dados = validar(base)
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_treino, i_teste = next(divisor.split(dados[VARIAVEIS], dados[ALVO], groups=dados["id_cliente"]))
    treino = dados.iloc[i_treino]
    teste = dados.iloc[i_teste]
    ids_treino = set(treino["id_cliente"])
    ids_teste = set(teste["id_cliente"])
    assert ids_treino.isdisjoint(ids_teste)
    modelo = criar_modelo()
    modelo.fit(treino[VARIAVEIS], treino[ALVO])
    real = teste[ALVO].to_numpy()
    previsto = modelo.predict(teste[VARIAVEIS])
    metricas = {
        "mae": float(mean_absolute_error(real, previsto)),
        "rmse": float(np.sqrt(mean_squared_error(real, previsto))),
        "r2": float(r2_score(real, previsto)),
        "registros_treino": len(treino),
        "registros_teste": len(teste),
        "clientes_treino": len(ids_treino),
        "clientes_teste": len(ids_teste),
        "clientes_em_ambos": len(ids_treino.intersection(ids_teste))
    }
    diagnosticos = teste[["id_cliente", "renda"]].copy()
    diagnosticos["previsto"] = previsto
    diagnosticos["erro_absoluto"] = abs(real - previsto)
    diagnosticos["erro_assinado"] = previsto - real
    return modelo, metricas, diagnosticos

def importancias(modelo):
    reg = modelo.regressor_
    nomes = reg.named_steps["tratamento"].get_feature_names_out()
    valores = reg.named_steps["floresta"].feature_importances_
    return pd.Series(valores, index=nomes).sort_values(ascending=False)

def prever(modelo, cliente):
    return modelo.predict(cliente[VARIAVEIS])
