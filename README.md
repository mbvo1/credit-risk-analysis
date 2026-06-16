# 🔍 DefaultLens
### Predição de Inadimplência com Ensemble Learning — Uma Análise Comparativa

> **Disciplina:** Aprendizado de Máquina  
> **Instituição:** CESAR School  
> **Autor:** Marcelo Bresani Victor de Oliveira — [@mbvo1](https://github.com/mbvo1)  
> **Google Sites:** https://sites.google.com/view/defaultlens/in%C3%ADcio


## Resumo

Este projeto implementa e compara cinco algoritmos de classificação para predição de inadimplência em crédito pessoal, utilizando o dataset **Home Credit Default Risk** (Kaggle, 307.511 registros). O trabalho replica a metodologia de Yang et al. (2025) — usando os mesmos dados e métricas — permitindo comparação direta com resultados validados por pares. O modelo AdaBoost otimizado obteve o melhor equilíbrio entre detecção de inadimplentes (Recall 0,423) e desempenho geral (F1 0,237), sendo selecionado para produção. A análise exploratória identificou forte desbalanceamento de classes (8,07% de inadimplentes), anomalias nos dados e multicolinearidade entre features, decisões que orientaram todo o pipeline de pré-processamento.


## Motivação e Contribuição

A predição de inadimplência é um problema central no setor financeiro, com impacto direto na sustentabilidade de instituições de crédito. Modelos de ML oferecem vantagem sobre métodos tradicionais por capturar interações não-lineares entre variáveis — algo que a análise exploratória deste trabalho confirma: nenhuma variável isolada separa bem as classes, tornando a abordagem multivariada indispensável.

Este trabalho contribui com:
- Replicação e extensão dos experimentos de Yang et al. (2025) com análise crítica das métricas
- Identificação e tratamento explícito de anomalias no dataset (valor sentinela em `DAYS_EMPLOYED`)
- Análise do trade-off precisão/recall entre os cinco modelos sob a ótica do negócio de crédito
- Pipeline completo de MLOps: rastreamento com MLflow, dashboard interativo e containerização Docker


## Dataset e Benchmark

**Dataset:** Home Credit Default Risk (Kaggle)  
**Registros:** 307.511 clientes | **Features:** 122 variáveis (demográficas, financeiras, contratuais)  
**Variável-alvo:** `TARGET` — 0 (adimplente) / 1 (inadimplente)  
**Desbalanceamento:** 91,93% adimplentes vs. 8,07% inadimplentes

**Paper benchmark (validado por pares):**
> Yang et al. (2025). *Interpretable Credit Default Prediction with Ensemble Learning and SHAP*. arXiv:2505.20815.

O benchmark foi escolhido por reportar resultados para exatamente os cinco modelos exigidos na disciplina (KNN, Árvore de Decisão, Random Forest, AdaBoost e MLP) com métricas diretamente comparáveis, permitindo validação rigorosa dos resultados.

---

## Resultados e Comparação com o Benchmark

Resultados obtidos neste trabalho (modelos otimizados, avaliados no conjunto de teste com distribuição real de 8,07%):

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Árvore de Decisão | 0,8463 | 0,1456 | 0,1857 | 0,1632 | 0,6099 |
| Random Forest | 0,8961 | 0,2243 | 0,1170 | 0,1538 | **0,6975** |
| AdaBoost ⭐ | 0,7799 | 0,1644 | 0,4228 | **0,2367** | 0,6920 |
| KNN | 0,3119 | 0,0943 | **0,8743** | 0,1702 | 0,6364 |
| MLP | 0,8318 | 0,1486 | 0,2290 | 0,1802 | 0,6390 |

Resultados reportados por Yang et al. (2025) para os mesmos modelos:

| Modelo | Acurácia | Precisão | Recall |
|---|---|---|---|
| Árvore de Decisão | 0,7350 | 0,6928 | 0,7162 |
| Random Forest | 0,7693 | 0,7551 | 0,7357 |
| KNN | 0,7215 | 0,6842 | 0,6938 |
| MLP | 0,7586 | 0,7482 | 0,7295 |
| AdaBoost | 0,7652 | 0,7510 | 0,7339 |

**Nota metodológica importante:** as diferenças entre os resultados são esperadas e decorrem de escolhas de pré-processamento distintas — em particular, o uso de SMOTE exclusivamente no treino (preservando a distribuição real no teste) e a avaliação com o conjunto de teste desbalanceado (8,07%), que reflete condições reais de deployment. O benchmark avalia em condições distintas. A comparação qualitativa das tendências entre modelos (qual performa melhor em qual métrica) é o ponto de convergência mais relevante.

⭐ **Modelo selecionado para produção:** AdaBoost — melhor F1 (0,237) e Recall substancial (0,423), representando o melhor equilíbrio entre detectar inadimplentes e manter precisão aceitável. Para instituições financeiras, o Recall é criticamente importante: um falso negativo (inadimplente classificado como adimplente) tem custo financeiro direto.

---

## Principais Descobertas da EDA

**1. Desbalanceamento severo (8,07% de inadimplentes)**  
Justifica o uso de SMOTE no treino e a escolha de métricas além da acurácia. Um classificador trivial que previsse "adimplente" para todos alcançaria 91,93% de acurácia sem qualquer utilidade prática.

**2. Variáveis EXT_SOURCE como principais preditores**  
EXT_SOURCE_1, 2 e 3 (scores de crédito de fontes externas) apresentaram as maiores correlações com inadimplência (-0,16 a -0,18). Este achado, obtido via análise de correlação, converge com os resultados SHAP de Yang et al. (2025) — validação independente da relevância dessas features.

**3. Anomalia em DAYS_EMPLOYED (valor sentinela)**  
365.243 dias (~1.000 anos) em 55.374 registros (~18% do dataset). Trata-se de um código para clientes sem vínculo empregatício. Tratamento aplicado: substituição por NaN + criação de flag binária `FLAG_SEM_EMPREGO`, preservando a informação como preditor.

**4. Natureza multivariada do risco**  
Mesmo a feature mais correlacionada (EXT_SOURCE_3, r = -0,18) tem correlação baixa individualmente. Nenhuma variável isolada separa bem as classes, confirmando que o risco de crédito emerge da combinação de múltiplos fatores — justificativa central para o uso de ML em vez de regras simples.

**5. Multicolinearidade identificada**  
AMT_CREDIT ↔ AMT_GOODS_PRICE (r = 0,99) e CNT_CHILDREN ↔ CNT_FAM_MEMBERS (r = 0,88). Relevante para modelos sensíveis a redundância de features.

---

## Pipeline Técnico

```
Dados brutos (307.511 × 122)
        ↓
EDA (8 visualizações + análise crítica de anomalias e distribuições)
        ↓
Pré-processamento
  ├── Tratamento da anomalia DAYS_EMPLOYED (sentinela → NaN + flag)
  ├── Descarte de colunas com >60% de faltantes (17 colunas removidas)
  ├── Imputação: mediana (numéricas) / moda (categóricas)
  ├── Codificação: Label Encoding (binárias + alta cardinalidade) +
  │              One-Hot Encoding com drop_first (demais categóricas)
  ├── Holdout estratificado 80/20 (random_state=42)
  ├── StandardScaler — fit exclusivo no treino (sem data leakage)
  └── SMOTE — aplicado exclusivamente no treino (teste preservado em 8,07%)
        ↓
Modelagem
  ├── 5 algoritmos: KNN, Árvore, Random Forest, AdaBoost, MLP
  ├── Otimização: Grid Search + Validação Cruzada (3 folds, scoring=AUC)
  └── Avaliação: Acurácia, Precisão, Recall, F1, AUC (no teste real)
        ↓
MLOps
  ├── Rastreamento: MLflow (backend SQLite)
  ├── Dashboard: Streamlit (previsão interativa por cliente)
  └── Containerização: Docker
```

---

## Limitações

- **Restrições computacionais:** grades de hiperparâmetros foram dimensionadas para viabilidade no hardware disponível. Buscas mais extensas poderiam melhorar os resultados, especialmente do Random Forest e AdaBoost.
- **KNN com amostra reduzida:** por limitações de memória e tempo, o KNN foi treinado com amostra de 50.000 pontos (do total de 452.296 balanceados). Resultados com o dataset completo poderiam diferir.
- **Tabelas auxiliares não utilizadas:** o dataset contém 9 tabelas adicionais (bureau, histórico de pagamentos, etc.). Este trabalho usa apenas a tabela principal (`application_train.csv`), alinhado ao benchmark de referência.
- **Discrepância CV vs. teste real:** métricas de validação cruzada (calculadas sobre dados SMOTE) são otimistas em relação ao teste real. Os valores reportados na tabela de resultados referem-se sempre ao conjunto de teste com distribuição real.

---

## Estrutura do Repositório

```
credit-risk-analysis/
├── data/               # Dados (não versionados — baixar via Kaggle)
├── notebooks/
│   ├── 01_eda.ipynb               # Análise Exploratória de Dados
│   └── 02_modelagem.ipynb         # Pré-processamento, modelagem e MLflow
├── app/
│   └── dashboard.py               # Dashboard interativo (Streamlit)
├── model/
│   ├── modelo.pkl                 # Modelo AdaBoost otimizado
│   └── scaler.pkl                 # StandardScaler
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Como Executar

### Pré-requisitos
- Python 3.12+
- Docker (para execução em container)
- Conta no Kaggle

### 1. Clonar o repositório
```bash
git clone https://github.com/mbvo1/credit-risk-analysis.git
cd credit-risk-analysis
```

### 2. Ambiente virtual
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Baixar os dados (Kaggle)
```bash
# Configure seu token: kaggle.com → Settings → API → Create New Token
export KAGGLE_API_TOKEN="seu_token"   # Linux/Mac
# $env:KAGGLE_API_TOKEN="seu_token"  # Windows PowerShell

kaggle competitions download -c home-credit-default-risk -p data
# Descompactar conforme seu SO
```

### 4. Executar notebooks
```
notebooks/01_eda.ipynb        → Análise Exploratória
notebooks/02_modelagem.ipynb  → Pré-processamento + Modelagem + MLflow
```

### 5. MLflow (visualizar experimentos)
```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db
# Acesse: http://127.0.0.1:5000
```

### 6. Dashboard (Streamlit)
```bash
streamlit run app/dashboard.py
# Acesse: http://localhost:8501
```

### 7. Docker (recomendado para reprodutibilidade)
```bash
docker build -t defaultlens .
docker run -p 8501:8501 defaultlens
# Acesse: http://localhost:8501
```

---

## Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Manipulação de dados | pandas, numpy |
| Visualização | matplotlib, seaborn |
| Machine Learning | scikit-learn, imbalanced-learn |
| Rastreamento | MLflow |
| Dashboard | Streamlit |
| Containerização | Docker |
| Versionamento | Git / GitHub |

---

## Referência

Yang et al. (2025). *Interpretable Credit Default Prediction with Ensemble Learning and SHAP*. arXiv:2505.20815. Disponível em: https://arxiv.org/abs/2505.20815
