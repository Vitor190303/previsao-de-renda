# 📊 Previsão de renda com Machine Learning

Projeto educacional de Ciência de Dados baseado no CRISP-DM, utilizando Random Forest e Streamlit.

**[Aplicação publicada](https://previsao-renda-vitor.streamlit.app)**

> ⚠️ Este modelo é experimental. Não utilizar para concessão de crédito, contratação, benefícios ou decisões de elegibilidade.

## Objetivo e dados

Estimar a renda a partir de informações sociodemográficas e profissionais. A base histórica possui **15.000 registros, 9.845 clientes distintos e 15 datas de referência** (janeiro de 2015 a março de 2016). O mesmo cliente pode aparecer em várias linhas.

Entradas: sexo, idade, tempo de emprego, tipo de renda, educação, estado civil, filhos, posse de veículo e imóvel, tipo de residência e quantidade de pessoas na residência. **Não entram no modelo:** `id_cliente`, `data_ref`, `Unnamed: 0` e `renda` (esta última é a variável alvo).

## Metodologia revisada

1. **Separar por cliente:** `GroupShuffleSplit(test_size=0.20, random_state=42)` mantém todos os registros de cada pessoa exclusivamente no treino **ou** teste. O percentual de 20% diz respeito ao número de clientes, não necessariamente de linhas.
2. **Aprender pré-processamento apenas no treino:** mediana numérica, moda categórica e `OneHotEncoder(handle_unknown="ignore")` ficam encapsulados em `Pipeline` / `ColumnTransformer`.
3. **Prever o log da renda:** `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` com `RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)`. Valores de renda devem ser positivos.
4. **Avaliar em reais:** calcular MAE, RMSE e R² no teste independente por cliente; o Streamlit exibe as métricas calculadas no carregamento, sem valores fixos.

**R² não é acurácia.** É uma métrica relativa à variabilidade observada no teste. **Importância da variável não é causa:** a redução de impureza nas árvores não estabelece relações causais.

## Resultados exploratórios prévios

Estes foram os resultados observados **em execuções manuais** antes da consolidação do pipeline atual. São valores de referência histórica, **não métricas calculadas automaticamente pela versão atual do pipeline**:

| Experimento | MAE | RMSE | R² |
|---|---:|---:|---:|
| Versão original, divisão aleatória | R$ 2.868,79 | R$ 5.625,31 | 0,3540 |
| Divisão por cliente, 100 árvores | R$ 2.985,38 | R$ 6.374,28 | 0,3187 |
| Divisão por cliente, 300 árvores, índice artificial | R$ 2.750,70 | R$ 6.003,75 | 0,3956 |
| Divisão por cliente, 300 árvores, sem índice | R$ 2.746,91 | R$ 6.013,90 | 0,3936 |

Os experimentos não compartilham necessariamente o mesmo teste/preparo: **diferenças de R² não identificam isoladamente o efeito causal de uma modificação**. Para obter os resultados da versão reproduzível, execute o app e veja as métricas nele calculadas.

### Limitações

Na análise exploratória houve tendência de superestimar rendas baixas e subestimar rendas altas. O erro absoluto médio pode ser grande, especialmente nas faixas mais elevadas. O diagnóstico por faixas usa **renda real conhecida retrospectivamente**, indisponível para um novo cliente. Variáveis sensíveis podem produzir disparidades entre grupos: o modelo não deve orientar decisões individuais de alto impacto.

A validação por cliente mede generalização para **clientes não vistos**. Uma validação temporal testaria outro objetivo (prever meses futuros) e permanece como extensão de pesquisa. Um mesmo cliente pode ter registros repetidos até dentro do mesmo mês.

## Como executar

```bash
git clone https://github.com/Vitor190303/previsao-de-renda.git
cd previsao-de-renda
python -m pip install -r requirements.txt
streamlit run app.py
```

O aplicativo lê `input/previsao_de_renda.csv` e treina um único pipeline, usado na avaliação e no formulário. Não depende dos arquivos legados em `output/Modelo/`, que continuam no repositório apenas como histórico. Na primeira execução, o treinamento pode usar mais recursos; o Streamlit usa cache nas demais sessões do processo.

## Organização

- `modelagem.py`: funções de validação, treino, pré-processamento, previsão e importâncias.
- `app.py`: interface Streamlit com métricas dinâmicas.
- `notebook_revisado.ipynb`: notebook principal e reproduzível, integrado ao pipeline de `modelagem.py`.
- `tests/test_modelagem.py`: testes de funcionamento com dados fictícios.
- `input/previsao_de_renda.csv`: dataset usado na execução.

O notebook anterior (`projeto-2.ipynb`) foi retirado da pasta principal. Seu desenvolvimento anterior pode ser consultado no histórico de versões (commits) do GitHub; a versão de referência para execução é `notebook_revisado.ipynb`.

**Autor:** Vitor Campos — projeto de estudos de Ciência de Dados.
