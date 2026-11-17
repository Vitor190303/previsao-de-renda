import numpy as np
import pandas as pd
from modelagem import VARIAVEIS, validar, treinar_e_avaliar, prever

def amostra(n=60):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id_cliente": np.repeat(np.arange(n // 3), 3),
        "data_ref": np.tile(["2015-01-01", "2015-02-01", "2015-03-01"], n // 3),
        "Unnamed: 0": np.arange(n),
        "sexo": rng.choice(["M", "F"], n),
        "tipo_renda": rng.choice(["Assalariado", "Pensionista"], n),
        "educacao": rng.choice(["Superior", "Secundário"], n),
        "estado_civil": rng.choice(["Solteiro", "Casado"], n),
        "tipo_residencia": rng.choice(["Casa", "Aluguel"], n),
        "posse_de_veiculo": rng.choice([True, False], n),
        "posse_de_imovel": rng.choice([True, False], n),
        "qtd_filhos": rng.integers(0, 3, n),
        "idade": rng.integers(18, 65, n),
        "tempo_emprego": np.where(np.arange(n) % 5 == 0, np.nan, rng.uniform(0, 20, n)),
        "qt_pessoas_residencia": rng.integers(1, 5, n),
        "renda": rng.uniform(800, 12000, n),
    })

def test_separacao_e_previsao():
    base = amostra()
    modelo, metricas, diagnosticos = treinar_e_avaliar(base)
    assert metricas["clientes_em_ambos"] == 0
    assert metricas["registros_treino"] + metricas["registros_teste"] == 60
    assert np.isfinite(metricas["mae"])
    assert len(diagnosticos) == metricas["registros_teste"]
    previsto = prever(modelo, base.iloc[[0]][VARIAVEIS])
    assert previsto[0] > 0 and np.isfinite(previsto[0])

def test_entrada_incompleta():
    try:
        validar(amostra().drop(columns=["id_cliente"]))
        assert False
    except ValueError:
        pass
