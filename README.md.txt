# 📊 Previsão de Renda com Machine Learning

Projeto de Ciência de Dados desenvolvido com o objetivo de analisar características de clientes e construir um modelo capaz de estimar sua renda.

O projeto foi desenvolvido seguindo as etapas do **CRISP-DM**, passando pela compreensão dos dados, análise exploratória, preparação, modelagem, avaliação e implantação do modelo através de uma aplicação interativa em **Streamlit**.

---

## 🎯 Objetivo do projeto

O objetivo é utilizar informações como:

- idade;
- sexo;
- tempo de emprego;
- tipo de renda;
- escolaridade;
- estado civil;
- quantidade de filhos;
- posse de veículo;
- posse de imóvel;
- tipo de residência;
- quantidade de pessoas na residência;

para estimar a **renda de um cliente** utilizando técnicas de Machine Learning.

---

## 📁 Base de dados

A base utilizada possui aproximadamente **15 mil registros de clientes**.

Entre as principais variáveis disponíveis estão características pessoais, profissionais e residenciais dos clientes, além da variável alvo:

`renda`

Durante a análise foi identificado que a renda possui uma distribuição bastante assimétrica, com presença de valores elevados.

Por esse motivo, foi utilizada uma transformação logarítmica da variável alvo durante a modelagem.

---

## 🔎 Análise exploratória

Durante a análise dos dados foram avaliados:

- distribuição da renda;
- renda por sexo;
- renda de clientes com e sem veículo;
- renda de clientes com e sem imóvel;
- correlação entre variáveis numéricas;
- relação entre tempo de emprego e renda;
- presença de valores extremos.

Entre os resultados encontrados, o **tempo de emprego apresentou uma das relações mais relevantes com a renda**.

---

## 🧹 Preparação dos dados

Antes da construção dos modelos foram realizadas etapas de preparação como:

- tratamento de valores ausentes;
- remoção de variáveis identificadoras;
- transformação de variáveis categóricas;
- criação de variáveis dummy;
- transformação logarítmica da renda;
- separação entre dados de treino e teste.

---

## 🤖 Modelos testados

Foram avaliados diferentes algoritmos de Machine Learning:

- Linear Regression;
- Ridge Regression;
- Lasso Regression;
- Decision Tree Regressor;
- Random Forest Regressor.

Também foram realizados testes de hiperparâmetros utilizando técnicas como **GridSearchCV**.

---

## 🏆 Modelo final

O modelo escolhido foi o:

### Random Forest Regressor

Resultados obtidos na base de teste:

| Métrica | Resultado |
|---|---:|
| MAE | R$ 2.868,79 |
| RMSE | R$ 5.625,31 |
| R² | 0,354 |

O modelo apresentou o melhor equilíbrio entre os algoritmos testados.

O R² indica que aproximadamente **35,4% da variabilidade da renda presente na base de teste foi explicada pelo modelo**.

Isso não significa que o modelo possui 35,4% de acurácia.

---

## 📈 Variáveis mais importantes

Segundo a Random Forest, algumas das variáveis com maior importância para as previsões foram:

- tempo de emprego;
- sexo;
- idade;
- tipo de renda;
- posse de imóvel;
- escolaridade.

A importância de uma variável dentro do modelo **não significa necessariamente uma relação de causa e efeito**.

---

## 🖥️ Aplicação Streamlit

Após a escolha do modelo final, foi criada uma aplicação em **Streamlit** para disponibilizar as análises e realizar novas previsões.

A aplicação possui duas áreas principais:

### 📊 Análises

Apresenta:

- quantidade de clientes;
- renda média;
- renda mediana;
- distribuição da renda por sexo;
- matriz de correlação;
- métricas do modelo;
- importância das variáveis.

### 🤖 Previsão de renda

Permite informar as características de um novo cliente e gerar uma estimativa de renda utilizando o modelo Random Forest treinado.

---

## 🗂️ Estrutura do projeto

```text
projeto-2/
│
├── app.py
├── projeto-2.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── input/
│   └── previsao_de_renda.csv
│
└── output/
    └── modelo/
        ├── random_forest.pkl
        ├── columns_modelo.pkl
        └── categorias.pkl
```

---

## 🚀 Como executar o projeto

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta:

```bash
cd projeto-2
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o Streamlit:

```bash
streamlit run app.py
```

---

## 🛠️ Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

---

## 📹 Demonstração

Uma demonstração da aplicação Streamlit será adicionada abaixo.

<!-- Aqui será inserido posteriormente o vídeo ou GIF do Streamlit -->

---

## 👨‍💻 Autor

**Vitor Campos**

Projeto desenvolvido como parte dos meus estudos em Ciência de Dados e construção de portfólio profissional.