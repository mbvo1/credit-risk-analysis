import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DefaultLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Estilo visual ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo e tipografia */
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    h1 { color: #ffffff; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #c9d1e0; font-size: 1.4rem; font-weight: 600; border-bottom: 1px solid #2a2d3a; padding-bottom: 8px; }
    h3 { color: #a0aec0; font-size: 1.1rem; }

    /* Card de destaque */
    .lens-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }

    /* Badge de risco */
    .risk-high { color: #fc8181; font-weight: 700; font-size: 1.8rem; }
    .risk-low  { color: #68d391; font-weight: 700; font-size: 1.8rem; }

    /* Barra de risco */
    .risk-bar-container {
        background: #2a2d3a;
        border-radius: 20px;
        height: 20px;
        width: 100%;
        margin: 12px 0;
        overflow: hidden;
    }

    /* Métrica */
    .metric-label { color: #718096; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e8eaf0; font-size: 1.4rem; font-weight: 700; }

    /* Header da marca */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .brand-tag {
        background: #3b4fd8;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Divider */
    hr { border-color: #2a2d3a; margin: 32px 0; }

    /* Esconder elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ─── Dados dos modelos (resultados já calculados) ─────────────────────────────
dados_modelos = {
    'Modelo': [
        'Árvore de Decisão', 'Random Forest',
        'AdaBoost', 'KNN', 'MLP'
    ],
    'Acurácia': [0.8463, 0.8961, 0.7799, 0.3119, 0.8318],
    'Precisão': [0.1456, 0.2243, 0.1644, 0.0943, 0.1486],
    'Recall':   [0.1857, 0.1170, 0.4228, 0.8743, 0.2290],
    'F1':       [0.1632, 0.1538, 0.2367, 0.1702, 0.1802],
    'AUC':      [0.6099, 0.6975, 0.6920, 0.6364, 0.6390],
}
df_modelos = pd.DataFrame(dados_modelos)

# ─── Carregar modelo e scaler ─────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    try:
        with open('model/modelo.pkl', 'rb') as f:
            modelo = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return modelo, scaler
    except FileNotFoundError:
        return None, None

modelo, scaler = carregar_modelo()

# ─── HEADER DA MARCA ──────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <span style="font-size:2rem;">🔍</span>
    <span class="brand-tag">ML · Crédito</span>
</div>
""", unsafe_allow_html=True)

st.title("DefaultLens")
st.markdown(
    "<p style='color:#718096; font-size:1.1rem; margin-top:-8px; margin-bottom:32px;'>"
    "A lente que enxerga o risco de inadimplência antes que ele aconteça."
    "</p>",
    unsafe_allow_html=True
)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Previsão", "📊 Modelos", "ℹ️ Sobre"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREVISÃO INTERATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Avalie o risco de um cliente")
    st.markdown(
        "<p style='color:#718096;'>Preencha os dados abaixo. "
        "O modelo AdaBoost (melhor equilíbrio entre detecção e precisão) "
        "retorna a probabilidade de inadimplência em tempo real.</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Perfil financeiro**")
        ext_source_1 = st.slider("Score externo 1 (EXT_SOURCE_1)", 0.0, 1.0, 0.5, 0.01,
                                  help="Score de crédito de fonte externa. Maior = melhor histórico.")
        ext_source_2 = st.slider("Score externo 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.5, 0.01)
        ext_source_3 = st.slider("Score externo 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.5, 0.01)
        amt_income   = st.number_input("Renda anual (R$)", 10000, 10000000, 150000, 5000)

    with col2:
        st.markdown("**Perfil do crédito**")
        amt_credit  = st.number_input("Valor do crédito (R$)", 10000, 5000000, 500000, 10000)
        amt_annuity = st.number_input("Parcela mensal (R$)", 1000, 200000, 25000, 1000)
        days_birth  = st.slider("Idade do cliente (anos)", 18, 70, 35)
        days_employed = st.slider("Tempo de emprego (anos)", 0, 40, 5)

    with col3:
        st.markdown("**Perfil pessoal**")
        code_gender     = st.selectbox("Gênero", ["Masculino", "Feminino"])
        flag_own_car    = st.selectbox("Possui carro?", ["Não", "Sim"])
        flag_own_realty = st.selectbox("Possui imóvel?", ["Não", "Sim"])
        cnt_children    = st.slider("Número de filhos", 0, 10, 0)

    st.markdown("---")
    prever = st.button("🔍 Analisar risco", use_container_width=True, type="primary")

    if prever:
        if modelo is None:
            st.warning(
                "⚠️ Modelo não encontrado. "
                "Certifique-se de que os arquivos `model/modelo.pkl` e "
                "`model/scaler.pkl` existem na pasta do projeto.",
                icon="⚠️"
            )
        else:
            # Montar vetor de features (137 colunas, preenchendo com 0 as não usadas)
            entrada = np.zeros(139)

            # Mapear os campos que o usuário preencheu para os índices corretos
            # (esses índices correspondem à ordem das colunas após o pré-processamento)
            mapa_features = {
                2:  ext_source_1,
                3:  ext_source_2,
                4:  ext_source_3,
                5:  amt_income,
                6:  amt_credit,
                7:  amt_annuity,
                8:  -days_birth * 365,         # converter para dias negativos
                9:  -days_employed * 365,
                10: 1 if code_gender == "Masculino" else 0,
                11: 1 if flag_own_car == "Sim" else 0,
                12: 1 if flag_own_realty == "Sim" else 0,
                13: cnt_children,
            }
            for idx, val in mapa_features.items():
                entrada[idx] = val

            # Padronizar e prever
            entrada_scaled = scaler.transform(entrada.reshape(1, -1))
            prob = modelo.predict_proba(entrada_scaled)[0][1]
            risco_pct = prob * 100

            # Mostrar resultado
            st.markdown("<br>", unsafe_allow_html=True)
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.markdown('<div class="lens-card">', unsafe_allow_html=True)
                st.markdown(
                    "<p class='metric-label'>Probabilidade de inadimplência</p>",
                    unsafe_allow_html=True
                )
                css_class = "risk-high" if risco_pct >= 50 else "risk-low"
                st.markdown(
                    f"<p class='{css_class}'>{risco_pct:.1f}%</p>",
                    unsafe_allow_html=True
                )
                veredicto = "⚠️ ALTO RISCO" if risco_pct >= 50 else "✅ BAIXO RISCO"
                st.markdown(f"**{veredicto}**")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_res2:
                st.markdown('<div class="lens-card">', unsafe_allow_html=True)
                st.markdown(
                    "<p class='metric-label'>Barra de risco</p>",
                    unsafe_allow_html=True
                )
                # Barra de risco colorida
                cor = f"linear-gradient(90deg, {'#fc8181' if risco_pct >= 50 else '#68d391'}, {'#feb2b2' if risco_pct >= 50 else '#9ae6b4'})"
                st.markdown(
                    f"""
                    <div class="risk-bar-container">
                        <div style="
                            width: {risco_pct}%;
                            height: 100%;
                            background: {cor};
                            border-radius: 20px;
                            transition: width 0.5s ease;
                        "></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#718096; font-size:0.75rem;">
                        <span>0% — Sem risco</span>
                        <span>50% — Limiar</span>
                        <span>100% — Risco total</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Fatores de risco identificados
                st.markdown("<br><p class='metric-label'>Fatores de risco identificados</p>", unsafe_allow_html=True)
                fatores = []
                if ext_source_2 < 0.3: fatores.append("🔴 Score externo 2 baixo")
                if ext_source_3 < 0.3: fatores.append("🔴 Score externo 3 baixo")
                if days_employed < 1:  fatores.append("🔴 Sem tempo de emprego")
                if amt_annuity / amt_income > 0.3: fatores.append("🟡 Parcela alta em relação à renda")
                if days_birth < 25:    fatores.append("🟡 Cliente jovem")
                if not fatores:        fatores.append("🟢 Nenhum fator de risco crítico detectado")
                for f in fatores:
                    st.markdown(f"- {f}")

                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARAÇÃO DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Comparação dos modelos")
    st.markdown(
        "<p style='color:#718096;'>Cinco modelos foram treinados, otimizados "
        "e avaliados com as mesmas métricas do benchmark (Yang et al., 2025).</p>",
        unsafe_allow_html=True
    )

    # Tabela de métricas
    st.dataframe(
        df_modelos.set_index('Modelo').style
        .highlight_max(axis=0, color='#1a3a2a', subset=['Acurácia', 'Precisão', 'Recall', 'F1', 'AUC'])
        .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    # Gráfico de barras comparativo
    st.markdown("### Comparação visual por métrica")
    metrica_sel = st.selectbox(
        "Selecione a métrica:",
        ['AUC', 'Recall', 'F1', 'Precisão', 'Acurácia']
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')

    valores = df_modelos[metrica_sel].values
    modelos_nomes = df_modelos['Modelo'].values
    cores = ['#3b4fd8' if v != max(valores) else '#68d391' for v in valores]

    bars = ax.barh(modelos_nomes, valores, color=cores, height=0.5)

    for bar, val in zip(bars, valores):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', color='#e8eaf0', fontsize=10)

    ax.set_xlim(0, max(valores) * 1.2)
    ax.set_xlabel(metrica_sel, color='#718096')
    ax.tick_params(colors='#a0aec0')
    ax.spines[:].set_color('#2a2d3a')
    melhor = mpatches.Patch(color='#68d391', label='Melhor resultado')
    outros = mpatches.Patch(color='#3b4fd8', label='Demais modelos')
    ax.legend(handles=[melhor, outros], facecolor='#1a1d2e',
              labelcolor='#a0aec0', framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Insight principal
    st.markdown('<div class="lens-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Conclusão analítica")
    st.markdown("""
Nenhum modelo domina em todas as métricas — cada um prioriza aspectos diferentes:

- **Random Forest** → melhor AUC (0,698): superior separação geral das classes
- **KNN** → melhor Recall (0,874): detecta ~87% dos inadimplentes, postura conservadora  
- **AdaBoost** → melhor F1 (0,237): melhor equilíbrio detecção/precisão — **escolhido para produção**

O trade-off entre Recall e Precisão reflete estratégias de negócio opostas:
um banco conservador prefere alto Recall (não deixar passar inadimplentes);
um banco agressivo prefere alta Precisão (não barrar bons clientes desnecessariamente).
""")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOBRE O PROJETO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Sobre o DefaultLens")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="lens-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Projeto")
        st.markdown("""
**Disciplina:** Aprendizado de Máquina  
**Instituição:** CESAR School  
**Autor:** Marcelo Bresani Victor de Oliveira  
**GitHub:** [@mbvo1](https://github.com/mbvo1)  

**Problema:** Predição de inadimplência em crédito pessoal  
**Dataset:** Home Credit Default Risk (Kaggle, 307.511 registros)  
**Benchmark:** Yang et al. (2025) — *Interpretable Credit Default Prediction with Ensemble Learning and SHAP*
""")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="lens-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Pipeline técnico")
        st.markdown("""
1. **EDA** — 8 visualizações com interpretação  
2. **Pré-processamento** — Limpeza, imputação, encoding, SMOTE  
3. **Modelagem** — 5 algoritmos: KNN, Árvore, Random Forest, AdaBoost, MLP  
4. **Otimização** — Grid Search + Validação Cruzada  
5. **Rastreamento** — MLflow (SQLite backend)  
6. **Interface** — Streamlit (este dashboard)  
7. **Containerização** — Docker  
""")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lens-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Por que DefaultLens?")
    st.markdown("""
O nome une dois conceitos centrais do projeto:

**Default** (inadimplência) — o fenômeno que o modelo aprende a detectar, 
presente em apenas 8% dos clientes mas com impacto financeiro desproporcional.

**Lens** (lente) — a metáfora da interpretabilidade: assim como uma lente revela 
detalhes invisíveis a olho nu, o modelo (e a análise SHAP) enxerga padrões de risco 
que análises superficiais não capturam. As variáveis EXT_SOURCE — as mais preditivas 
descobertas na análise exploratória — são essa lente sobre o histórico de crédito.
""")
    st.markdown('</div>', unsafe_allow_html=True)