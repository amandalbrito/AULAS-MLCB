

EXERCÍCIO 2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import IPython.display as display

# 1. Criar o Dataset
datas = pd.date_range(start='2023-01-01', periods=365, freq='D')
temperaturas = 20 + 5 * np.random.normal(size=365)
dias_uteis = datas.dayofweek < 5
consumo = 50 + temperaturas * 2 + dias_uteis * 30 + np.random.randn(365) * 10

data = {
    'data': datas,
    'temperatura_media': temperaturas,
    'dia_util': dias_uteis,
    'consumo_energia_kwh': consumo
}

df = pd.DataFrame(data)
display.display(df.head())

# 2. Engenharia de Atributos
df['mes'] = df['data'].dt.month
df['dia_da_semana'] = df['data'].dt.dayofweek
df['dia_do_ano'] = df['data'].dt.dayofyear
df = df.drop('data', axis=1)
display.display(df.head())

# 3. Preparação dos Dados
X = df.drop('consumo_energia_kwh', axis=1)
y = df['consumo_energia_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelagem
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 5. Treinamento do modelo
model.fit(X_train, y_train)

# 6. Avaliação do modelo
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Absolute Error (MAE): {mae:.2f} kWh')
print(f'R-squared (R²): {r2:.2f}')

# 7. Interpretação dos resultados
print(f"""\nInterpretação do MAE:
O Erro Absoluto Médio (MAE) calculado foi de aproximadamente {mae:.2f} kWh.""")
