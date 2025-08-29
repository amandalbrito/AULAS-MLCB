#### TAREFA 1 :  Treine o modelo abaixo colocando mais 4 ITENS ADICIONAIS e rode sem erros no VSCODE ou no Colab - Faça teste após o treinamento ###

#Para começar, instale o scikit-learn = pip install scikit-learn

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Função de pré-processamento
def limpar_texto(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    texto = re.sub(r'\d+', '', texto)  # Remove números
    texto = texto.strip()  # Remove espaços extras
    return texto

# 1. Conjunto de dados (mensagens + rótulos)
mensagens = [
    "Quero fazer um pedido",
    "Preciso falar com o suporte",
    "Quais promoções vocês têm hoje?",
    "Qual o horário de funcionamento?",
    "Meu produto veio com defeito",
    "Posso pagar com cartão de crédito?",
    "Qual é o valor?",
    "Quais são os serviços oferecidos?",
    "Posso falar com uma pessoa?",
    "Eu vou chamar a polícia",
    "Uma pessoa é um ser humano"
]
rotulos = ["pedido", "suporte", "promoção", "informação", "suporte", "pagamento", "pagamento", "serviços", "indivíduos","emergência", "indivíduos"]

# 2. Pré-processamento das mensagens
mensagens_limpas = [limpar_texto(m) for m in mensagens]

# 3. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensagens_limpas)

# 4. Treinamento do modelo
modelo = MultinomialNB()
modelo.fit(X, rotulos)

# 5. Interação com o usuário
while True:
    nova_mensagem = input("\nDigite uma mensagem (ou 'sair' para encerrar): ")
    if nova_mensagem.lower() == "sair":
        break
    nova_mensagem_limpa = limpar_texto(nova_mensagem)
    X_novo = vectorizer.transform([nova_mensagem_limpa])
    predicao = modelo.predict(X_novo)
    print(f"Intenção prevista: {predicao[0]}")





#### TAREFA 2 :  Criar um classificador de mensagens para um bot de atendimento acadêmico - rode sem erros no VSCODE  ou no Colab ###

# Criar um classificador de mensagens para um bot de atendimento acadêmico.
# Instruções:
# 1. Crie uma lista de frases (ex: dúvidas sobre matrícula, notas, eventos, biblioteca)
# 2. Crie a lista de rótulos correspondentes
# 3. Vetorize as frases com CountVectorizer
# 4. Treine um modelo Naive Bayes ou outro de sua escolha
# 5. Teste com uma nova frase e imprima o resultado

# início código
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re


# Função de pré-processamento
def limpar_texto(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    texto = re.sub(r'\d+', '', texto)  # Remove números
    texto = texto.strip()  # Remove espaços extras
    return texto

# 1. Dataset
frases = [
    "Consulta de matricula",
    "Consultar as mensalidades",
    "Meu boleto está pago?",
    "Quantas horas complementares eu preciso?"

]
rotulos = ["Matrícula","Mensalidades", "Mensalidades", "Horas complementares"
]

# 2. Pré-processamento das mensagens
mensagens_limpas = [limpar_texto(m) for m in frases]

# 3. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensagens_limpas)

# 3. Modelo
modelo = MultinomialNB()
modelo.fit(X, rotulos)

# 4. Previsão
while True:
    nova_mensagem = input("\nComo posso ajudar? ")
    if nova_mensagem.lower() == "sair":
        break
    nova_mensagem_limpa = limpar_texto(nova_mensagem)
    X_novo = vectorizer.transform([nova_mensagem_limpa])
    predicao = modelo.predict(X_novo)
    print(f"Intenção prevista: {predicao[0]}")



#### TAREFA 3 :  Criar um classificador de Previsão de Tempo de Entrega de Pizza - rode sem erros no VSCODE  ou no Colab###
# ------------------------------------------------------------------------------
#criar um modelo que prevê o tempo de entrega (em minutos) com base na distância (em km) e no número de pizzas no pedido.
# ------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
import numpy as np

print("\n--- 1.2 Exercício para Alunos (Supervisionado) ---")

# Dados de Treino: [distancia_km, numero_de_pizzas]
dados_entregas = np.array([
    [5, 2],   # 5 km, 2 pizzas
    [2, 1],   # 2 km, 1 pizza
    [10, 4],  # 10 km, 4 pizzas
    [7, 3],   # 7 km, 3 pizzas
    [1, 1]    # 1 km, 1 pizza
])

# Rótulos: Tempo de entrega em minutos
tempos_entrega = np.array([30, 15, 55, 40, 10])

# TODO: Crie uma instância do modelo LinearRegression.
modelo_entrega = LinearRegression()

# TODO: Treine o modelo usando os dados de entregas e os tempos.
modelo_entrega.fit(dados_entregas, tempos_entrega)

# TODO: Faça a previsão para um novo pedido: 8 km de distância e 2 pizzas.
pedido_novo = np.array([[8, 2]])
tempo_previsto = modelo_entrega.predict(pedido_novo)

print(f"Tempo de entrega previsto para o novo pedido: {tempo_previsto[0]:.2f} minutos")
