import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuração da página Web
st.set_page_config(page_title="Preditor Copa 2026", page_icon="⚽", layout="centered")

# ==========================================
# 1. LISTA OFICIAL DA COPA 2026 (48 Seleções)
# ==========================================
SELECOES_2026 = [
    # América do Norte
    "Canada", "USA", "Mexico", "Curaçao", "Haiti", "Panama",
    # América do Sul
    "Argentina", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay",
    # Europa
    "Germany", "Austria", "Belgium", "Bosnia and Herzegovina", "Croatia", 
    "Scotland", "Spain", "France", "England", "Netherlands", "Norway", 
    "Portugal", "Czechia", "Sweden", "Switzerland", "Turkey",
    # Ásia
    "Saudi Arabia", "Australia", "Qatar", "Korea Republic", "IR Iran", 
    "Iraq", "Japan", "Jordan", "Uzbekistan",
    # África
    "South Africa", "Algeria", "Cape Verde Islands", "Côte d'Ivoire", "Egypt", 
    "Ghana", "Morocco", "Congo DR", "Senegal", "Tunisia",
    # Oceania
    "New Zealand"
]

# ==========================================
# 2. CARREGAMENTO DOS DADOS RECENTES (Cache)
# ==========================================
@st.cache_data
def carregar_dados_recentes():
    df_jogos = pd.read_csv('dataset/results.csv')
    df_jogos['date'] = pd.to_datetime(df_jogos['date'])
    df_jogos = df_jogos[df_jogos['date'].dt.year >= 2000].copy() 
    
    dicionario_paises = {
        'United States': 'USA', 'South Korea': 'Korea Republic', 
        'Iran': 'IR Iran', 'Czech Republic': 'Czechia'
    }
    
    df_jogos = df_jogos.rename(columns={'date': 'data', 'home_team': 'mandante', 'away_team': 'visitante', 'home_score': 'gols_mandante', 'away_score': 'gols_visitante'})
    df_jogos['mandante'] = df_jogos['mandante'].replace(dicionario_paises)
    df_jogos['visitante'] = df_jogos['visitante'].replace(dicionario_paises)

    df_ranking = pd.read_csv('dataset/fifa_ranking.csv')
    df_ranking['rank_date'] = pd.to_datetime(df_ranking['rank_date'])
    df_ranking = df_ranking[['rank_date', 'country_full', 'rank']]
    df_ranking['country_full'] = df_ranking['country_full'].replace(dicionario_paises)

    df_jogos = df_jogos.sort_values('data')
    df_ranking = df_ranking.sort_values('rank_date')

    # Cruzamento Temporal
    df = pd.merge_asof(df_jogos, df_ranking, left_on='data', right_on='rank_date', left_by='mandante', right_by='country_full', direction='backward').rename(columns={'rank': 'rank_mandante'}).drop(columns=['rank_date', 'country_full'])
    df = pd.merge_asof(df, df_ranking, left_on='data', right_on='rank_date', left_by='visitante', right_by='country_full', direction='backward').rename(columns={'rank': 'rank_visitante'}).drop(columns=['rank_date', 'country_full'])

    # Organizando features
    df_mandante = df[['data', 'mandante', 'gols_mandante', 'gols_visitante']].rename(columns={'mandante': 'time', 'gols_mandante': 'feitos', 'gols_visitante': 'sofridos'})
    df_visitante = df[['data', 'visitante', 'gols_visitante', 'gols_mandante']].rename(columns={'visitante': 'time', 'gols_visitante': 'feitos', 'gols_mandante': 'sofridos'})
    
    df_times = pd.concat([df_mandante, df_visitante]).sort_values('data')
    df_times[['media_ataque_5', 'media_defesa_5']] = df_times.groupby('time')[['feitos', 'sofridos']].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    df = pd.merge(df, df_times[['data', 'time', 'media_ataque_5', 'media_defesa_5']], left_on=['data', 'mandante'], right_on=['data', 'time'], how='left').rename(columns={'media_ataque_5': 'ataq_mand', 'media_defesa_5': 'def_mand'}).drop(columns=['time'])
    df = pd.merge(df, df_times[['data', 'time', 'media_ataque_5', 'media_defesa_5']], left_on=['data', 'visitante'], right_on=['data', 'time'], how='left').rename(columns={'media_ataque_5': 'ataq_vis', 'media_defesa_5': 'def_vis'}).drop(columns=['time'])

    df = df.dropna(subset=['rank_mandante', 'rank_visitante', 'ataq_mand', 'def_mand', 'ataq_vis', 'def_vis'])
    
    # === A MÁGICA DO FILTRO ACONTECE AQUI ===
    # Pega todos os países que existem no CSV
    todos_paises_csv = set(df['mandante'].unique()) | set(df['visitante'].unique())
    # Filtra mantendo apenas os que estão na nossa lista oficial de 2026
    paises_filtrados = [pais for pais in SELECOES_2026 if pais in todos_paises_csv]
    # Organiza em ordem alfabética para o SelectBox
    paises_filtrados.sort()
    
    return df, paises_filtrados

# ==========================================
# 3. CARREGAMENTO DOS MODELOS DE IA
# ==========================================
@st.cache_resource
def carregar_arquivos_ia():
    try:
        modelo_rf = joblib.load('modelo/randon_forest.pkl')
        modelo_lr = joblib.load('modelo/logistic_regression.pkl')
        scaler = joblib.load('modelo/scaler_logistic_regression.pkl')
        return modelo_rf, modelo_lr, scaler
    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado: {e.filename}. Verifique se os arquivos .pkl estão na mesma pasta do aplicacao.py.")
        st.stop()

# ==========================================
# 4. INTERFACE VISUAL PRINCIPAL
# ==========================================
st.title("🏆 IA: Previsões Copa 2026")
st.write("Selecione o motor de Inteligência Artificial e simule confrontos usando estatísticas atualizadas.")

with st.spinner('Inicializando sistema...'):
    df, lista_paises = carregar_dados_recentes()
    modelo_rf, modelo_lr, scaler = carregar_arquivos_ia()

# Opção de selecionar o modelo
st.markdown("### ⚙️ Configuração do Modelo")
modelo_escolhido = st.radio(
    "Escolha o algoritmo de predição:",
    ("Random Forest (Árvores de Decisão)", "Regressão Logística (Estatística)"),
    horizontal=True
)

st.divider()

# Seleção das equipes
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    mandante = st.selectbox("Seleção 1 (Mandante)", lista_paises, index=lista_paises.index("Brazil") if "Brazil" in lista_paises else 0)

with col2:
    st.markdown("<h3 style='text-align: center; margin-top: 30px;'>VS</h3>", unsafe_allow_html=True)

with col3:
    visitante = st.selectbox("Seleção 2 (Visitante)", lista_paises, index=lista_paises.index("Morocco") if "Morocco" in lista_paises else 1)


# ==========================================
# 5. LÓGICA DE INFERÊNCIA E EXIBIÇÃO
# ==========================================
if st.button("Simular Confronto", use_container_width=True):
    if mandante == visitante:
        st.error("Por favor, selecione equipes diferentes!")
    else:
        try:
            # 1. Pega as métricas brutas mais recentes
            ultimo_m = df[df['mandante'] == mandante].iloc[-1]
            ultimo_v = df[df['visitante'] == visitante].iloc[-1]
            
            cenario_bruto = pd.DataFrame({
                'dif_ranking': [ultimo_m['rank_mandante'] - ultimo_v['rank_visitante']],
                'ataq_mand': [ultimo_m['ataq_mand']],
                'def_mand': [ultimo_m['def_mand']],
                'ataq_vis': [ultimo_v['ataq_vis']],
                'def_vis': [ultimo_v['def_vis']]
            })
            
            # 2. Roteamento do Modelo
            if modelo_escolhido == "Random Forest (Árvores de Decisão)":
                # RF usa os dados puros
                classe = modelo_rf.predict(cenario_bruto)[0]
                probs = modelo_rf.predict_proba(cenario_bruto)[0]
            else:
                # Regressão Logística OBRIGA a normalização dos dados
                cenario_normalizado = scaler.transform(cenario_bruto)
                classe = modelo_lr.predict(cenario_normalizado)[0]
                probs = modelo_lr.predict_proba(cenario_normalizado)[0]
            
            mapa = {0: visitante, 1: "Empate", 2: mandante}
            vencedor = mapa[classe]

            # 3. Desenha os resultados na tela
            st.divider()
            st.subheader(f"Veredicto IA: **{vencedor.upper()}**")
            
            st.write(f"**Vitória do {mandante}:** {probs[2]*100:.1f}%")
            st.progress(float(probs[2]))
            
            st.write(f"**Empate:** {probs[1]*100:.1f}%")
            st.progress(float(probs[1]))
            
            st.write(f"**Vitória do {visitante}:** {probs[0]*100:.1f}%")
            st.progress(float(probs[0]))
            
            # 4. Transparência das estatísticas (Debugging Visual)
            st.divider()
            st.write("📊 **Base de Dados Atual:**")
            
            col_m, col_v = st.columns(2)
            with col_m:
                st.metric(label=f"Rank FIFA - {mandante}", value=int(ultimo_m['rank_mandante']))
                st.metric(label="Ataque (Média Gols)", value=f"{ultimo_m['ataq_mand']:.2f}")
                st.metric(label="Defesa (Média Sofridos)", value=f"{ultimo_m['def_mand']:.2f}")
            with col_v:
                st.metric(label=f"Rank FIFA - {visitante}", value=int(ultimo_v['rank_visitante']))
                st.metric(label="Ataque (Média Gols)", value=f"{ultimo_v['ataq_vis']:.2f}")
                st.metric(label="Defesa (Média Sofridos)", value=f"{ultimo_v['def_vis']:.2f}")
                
        except IndexError:
            st.error(f"Falta de dados históricos recentes para analisar {mandante} x {visitante}.")
