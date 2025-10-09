EXERCÍCIO 1

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


np.random.seed(42)
n = 400  # número de registros

dados = pd.DataFrame({
    "idade": np.random.randint(25, 66, size=n),
    "salario_anual": np.random.randint(20000, 200000, size=n).astype(float),
    "anos_empregado": np.random.randint(0, 31, size=n),
    "valor_emprestimo": np.random.randint(5000, 150000, size=n),
    "tipo_moradia": np.random.choice(["Aluguel", "Propria", "Financiada"], size=n)
})

mask = np.random.choice([True, False], size=n, p=[0.1, 0.9])
dados.loc[mask, "salario_anual"] = np.nan

dados["proporcao_emprestimo_salario"] = dados["valor_emprestimo"] / dados["salario_anual"]


dados["risco_inadimplencia"] = (
    (dados["proporcao_emprestimo_salario"] > 1.0) |  # empréstimo maior que o salário
    ((dados["valor_emprestimo"] > 80000) & (dados["anos_empregado"] < 3)) |
    ((dados["salario_anual"] < 40000) & (dados["valor_emprestimo"] > 30000))
).astype(int)


X = dados.drop("risco_inadimplencia", axis=1)
y = dados["risco_inadimplencia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

num_cols = ["idade", "salario_anual", "anos_empregado", "valor_emprestimo", "proporcao_emprestimo_salario"]
cat_cols = ["tipo_moradia"]

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


modelo = Pipeline(steps=[
    ("preprocessamento", preprocessor),
    ("classificador", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])


modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

print("🔹 Acurácia:", round(accuracy_score(y_test, y_pred), 3))
print("\n🔹 Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("🔹 ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))


novo_cliente = pd.DataFrame({
    "idade": [35],
    "salario_anual": [10000],
    "anos_empregado": [8],
    "valor_emprestimo": [40000],
    "tipo_moradia": ["Propria"]
})


novo_cliente["proporcao_emprestimo_salario"] = novo_cliente["valor_emprestimo"] / novo_cliente["salario_anual"]


risco_previsto = modelo.predict(novo_cliente)[0]
probabilidade = modelo.predict_proba(novo_cliente)[0][1]

print("\n🔹 Previsão para novo cliente:")
print("Risco:", "Alto Risco" if risco_previsto == 1 else "Baixo Risco")
print("Probabilidade de inadimplência:", f"{probabilidade:.2%}")

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

EXERCÍCIO 3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

dados = {
    "texto": [
        "oferta imperdível clique aqui agora",
        "ganhe dinheiro fácil e rápido",
        "relatório de vendas anexo",
        "oi, tudo bem? reunião amanhã",
        "promoção exclusiva aproveite já",
        "parabéns, você ganhou um prêmio",
        "bom dia, segue o relatório solicitado",
        "confirmação de pagamento recebida",
        "aumente sua renda sem sair de casa",
        "convite para entrevista de emprego",
        "compre agora com desconto especial",
        "você foi selecionado para um sorteio",
        "pagamento da fatura confirmado",
        "resumo de atividades do mês",
        "ganhe um cupom de 50% de desconto agora",
        "envio do contrato assinado",
        "última chance para participar",
        "relatório financeiro mensal",
        "dinheiro rápido e garantido",
        "lembrete: reunião às 10h"
    ],
    "categoria": [
        "spam", "spam", "ham", "ham", "spam",
        "spam", "ham", "ham", "spam", "ham",
        "spam", "spam", "ham", "ham", "spam",
        "ham", "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(dados)

X = df["texto"]
y = df["categoria"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelo = Pipeline([
    ("vetorizador", TfidfVectorizer()),
    ("classificador", MultinomialNB())
])

modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)

print(f"Acurácia: {accuracy_score(y_teste, y_pred):.3f}\n")
print("🔹 Relatório de Classificação:")
print(classification_report(y_teste, y_pred, zero_division=0))


novos_emails = [
    "ganhe um iPhone agora mesmo",
    "segue o contrato atualizado",
    "investimento com retorno garantido",
    "reunião confirmada às 15h"
]

previsoes = modelo.predict(novos_emails)

print("\n🔹 Previsões para novos e-mails:")
for email, pred in zip(novos_emails, previsoes):
    print(f"'{email}' → {pred}")

EXERCÍCIO 4

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar o dataset
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset Iris carregado.")

# 2. Preparação dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos: Treino ({len(X_train)} amostras), Teste ({len(X_test)} amostras)")

# 3. Modelagem
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
print("Pipeline do modelo criado (StandardScaler + KNeighborsClassifier).")

# 4. Treinamento do modelo
model.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

# 5. Avaliação do modelo
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nAcurácia no conjunto de teste: {accuracy:.2f}')

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('\nRelatório de Classificação:')
print(report)

EXERCÍCIO 5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=";")


X = df.drop("quality", axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n Resultados do Modelo:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")

print("\n Interpretação:")
print(f"O Erro Absoluto Médio (MAE) indica que, em média, o modelo erra a nota do vinho em cerca de {mae:.2f} pontos.")
print(f"O R² = {r2:.2f} mostra o quanto do comportamento da qualidade o modelo consegue explicar:")
print("- Valores próximos de 1 indicam excelente ajuste.")

