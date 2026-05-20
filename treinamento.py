import pandas as pd
import numpy as np

#Randon Forest
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

#Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

import warnings
warnings.filterwarnings('ignore')

# 1. Carregando Jogos
df_jogos = pd.read_csv('dataset/results.csv')
df_jogos['date'] = pd.to_datetime(df_jogos['date'])
df_jogos = df_jogos[df_jogos['date'].dt.year >= 2000].copy()

df_jogos = df_jogos.rename(columns={
    'date': 'data', 'home_team': 'mandante', 'away_team': 'visitante',
    'home_score': 'gols_mandante', 'away_score': 'gols_visitante'
})

# Dicionário de padronização
dicionario_paises = {
    'United States': 'USA', 'South Korea': 'Korea Republic',
    'Iran': 'IR Iran', 'Czech Republic': 'Czechia'
}
df_jogos['mandante'] = df_jogos['mandante'].replace(dicionario_paises)
df_jogos['visitante'] = df_jogos['visitante'].replace(dicionario_paises)

# 2. Carregando Rankings
df_ranking = pd.read_csv('dataset/fifa_ranking.csv')
df_ranking['rank_date'] = pd.to_datetime(df_ranking['rank_date'])
df_ranking = df_ranking[['rank_date', 'country_full', 'rank']]
df_ranking['country_full'] = df_ranking['country_full'].replace(dicionario_paises)

# Ordenação obrigatória para os próximos passos
df_jogos = df_jogos.sort_values('data')
df_ranking = df_ranking.sort_values('rank_date')

df = pd.merge_asof(df_jogos, df_ranking, left_on='data', right_on='rank_date',
                   left_by='mandante', right_by='country_full', direction='backward')
df = df.rename(columns={'rank': 'rank_mandante'}).drop(columns=['rank_date', 'country_full'])

df = pd.merge_asof(df, df_ranking, left_on='data', right_on='rank_date',
                   left_by='visitante', right_by='country_full', direction='backward')
df = df.rename(columns={'rank': 'rank_visitante'}).drop(columns=['rank_date', 'country_full'])

df['dif_ranking'] = df['rank_mandante'] - df['rank_visitante']

# 2. Cálculo de Ataque/Defesa
df['resultado'] = np.where(df['gols_mandante'] > df['gols_visitante'], 2,
                  np.where(df['gols_mandante'] == df['gols_visitante'], 1, 0))

df_mandante = df[['data', 'mandante', 'gols_mandante', 'gols_visitante']].rename(
    columns={'mandante': 'time', 'gols_mandante': 'feitos', 'gols_visitante': 'sofridos'})
df_visitante = df[['data', 'visitante', 'gols_visitante', 'gols_mandante']].rename(
    columns={'visitante': 'time', 'gols_visitante': 'feitos', 'gols_mandante': 'sofridos'})

df_times = pd.concat([df_mandante, df_visitante]).sort_values('data')

df_times[['media_ataque_5', 'media_defesa_5']] = (
    df_times.groupby('time')[['feitos', 'sofridos']]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

df = pd.merge(df, df_times[['data', 'time', 'media_ataque_5', 'media_defesa_5']],
              left_on=['data', 'mandante'], right_on=['data', 'time'], how='left') \
       .rename(columns={'media_ataque_5': 'ataq_mand', 'media_defesa_5': 'def_mand'}).drop(columns=['time'])

df = pd.merge(df, df_times[['data', 'time', 'media_ataque_5', 'media_defesa_5']],
              left_on=['data', 'visitante'], right_on=['data', 'time'], how='left') \
       .rename(columns={'media_ataque_5': 'ataq_vis', 'media_defesa_5': 'def_vis'}).drop(columns=['time'])

# Removemos linhas onde faltam dados (ex: o primeiríssimo jogo de uma seleção)
df = df.dropna(subset=['dif_ranking', 'ataq_mand', 'def_mand', 'ataq_vis', 'def_vis'])

print("Features calculadas! O dataset está pronto para o modelo.")
df[['data', 'mandante', 'visitante', 'dif_ranking', 'ataq_mand', 'def_mand']].tail() # Espia os jogos mais recentes

features = ['dif_ranking', 'ataq_mand', 'def_mand', 'ataq_vis', 'def_vis']
X = df[features]
y = df['resultado']

#Modelo Randon Forest
print("Balanceando os dados com SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

print("Treinando o Random Forest (isso pode levar alguns segundos)...")
modelo = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
modelo.fit(X_bal, y_bal)

# Salvar o modelo
joblib.dump(modelo, 'modelo/randon_forest.pkl')

#Modelo Regressão Logística Multinomial
# 1. NORMALIZAÇÃO: Coloca todas as métricas na mesma escala matemática
print("Normalizando as métricas...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. BALANCEAMENTO (Continuamos usando o SMOTE para os empates)
print("Balanceando os dados com SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# 3. TREINAMENTO
print("Treinando a Regressão Logística ...")
# max_iter=1000 garante que a equação tenha tempo de fazer os cálculos até o fim
# modelo_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
modelo_lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
modelo_lr.fit(X_bal, y_bal)

print("Modelo treinado com sucesso!")

joblib.dump(modelo_lr, 'modelo/logistic_regression.pkl')
joblib.dump(scaler, 'modelo/scaler_logistic_regression.pkl')