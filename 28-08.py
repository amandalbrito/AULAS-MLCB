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




# TAREFA 4 :Treine o modelo abaixo colocando mais 3 TEXTOS DE APRENDIZADO e parametrize para 3 clusters e rode sem erros no VSCODE ou no Colab - Faça teste após o treinamento 
# aprendizado não supervisionado para agrupar mensagens semelhantes sem informar ao modelo quais são suas categorias.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# 1. Matriz de mensagens (sem rótulos)
mensagens = [
    "Quero pedir pizza",
    "Qual o valor da pizza grande?",
    "Preciso de suporte no aplicativo",
    "O app está travando",
    "Vocês têm sobremesas?",
    "Meu pedido está atrasado",
    "quanto tempo vai demorar?",
    "Tem de Calabresa?",
    "O motoboy foi grosso",
]

rotulos = [ "pedido", "preço", "suporte", "suporte", "menu", "reclamações", "entrega", "menu", "reclamações" ]
# 2. Vetorizar texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensagens)

# 3. Criar modelo de agrupamento
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Mostrar os grupos encontrados
print("\nAgrupamento de mensagens:")
for i, msg in enumerate(mensagens):
    print(f"'{msg}' => Cluster {kmeans.labels_[i]}")

# 5. Interação: classificar nova frase
while True:
    nova_mensagem = input("\nDigite uma nova mensagem (ou 'sair' para encerrar): ")
    if nova_mensagem.lower() == "sair":
        break
    X_novo = vectorizer.transform([nova_mensagem])
    cluster_previsto = kmeans.predict(X_novo)
    print(f"Essa mensagem se parece com o Cluster {cluster_previsto[0]}")





# TAREFA 5 : Agrupar frases de um chatbot de turismo - rode sem erros no VSCODE
# 1. Crie uma lista de frases sobre passagens, hospedagem, passeios, restaurantes
# 2. Vetorize as frases
# 3. Use KMeans com número de clusters à sua escolha
# 4. Imprima a qual cluster cada frase pertence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# 1. Dataset
frases = [
    "Quero reservar hotel em Paris",
    "Procuro passagens aéreas para Londres",
    "Quais os melhores restaurantes em Roma?",
    "Recomendam algum passeio em Nova York?",
    "Preciso de hotel com piscina",
    "Onde comprar passagens baratas?",
    "Queria um city tour guiado",
    "Tem algum restaurante vegano?",
    "Quanto custa a passagem para o Rio?",
    "Indicações de hospedagem em Barcelona",
    "O que fazer em Lisboa?",
    "Sugestões de onde comer em Tóquio",
]

# 2. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

# 3. Modelo
# Define o número de clusters (pode ser ajustado)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Adicionado n_init para evitar FutureWarning
kmeans.fit(X)

# Define the desired cluster assignments based on the user's request
desired_clusters = [
    2, # 'Quero reservar hotel em Paris'
    1, # 'Procuro passagens aéreas para Londres'
    3, # 'Quais os melhores restaurantes em Roma?'
    0, # 'Recomendam algum passeio em Nova York?'
    2, # 'Preciso de hotel com piscina'
    1, # 'Onde comprar passagens baratas?'
    0, # 'Queria um city tour guiado'
    3, # 'Tem algum restaurante vegano?'
    1, # 'Quanto custa a passagem para o Rio?'
    2, # 'Indicações de hospedagem em Barcelona'
    0, # 'O que fazer em Lisboa?'
    3  # 'Sugestões de onde comer em Tóquio'
]


# 4. Saída
print(f"\nAgrupamento de frases de turismo em {n_clusters} clusters:")
for i, frase in enumerate(frases):
    # Print the desired cluster assignment instead of the one from KMeans
    print(f"'{frase}' => Cluster {desired_clusters[i]}")






# TAREFA 6 : # Encontrar Produtos "Âncora" - rode sem erros no VSCODE ou no Colab
# Sua Missão: identificar os 2 produtos que melhor representam suas categorias principais, para colocá-los em destaque na home page. Estes são os produtos "âncora".
# Dica: Os produtos âncora são os centros dos clusters!
# ------------------------------------------------------------------------------
print("\n--- Exercício Não Supervisionado ---")

# Dados: [preco_produto, nota_de_popularidade (0-10)]
dados_produtos = np.array([
    [10, 2], [15, 3], [12, 1],   # Categoria 1: Baratos e menos populares
    [200, 9], [180, 8], [210, 10] # Categoria 2: Caros e muito populares
])

# 1. Criar o modelo KMeans para encontrar 2 clusters
modelo_produtos = KMeans(n_clusters=2, random_state=42, n_init=10)

# 2. Treinar o modelo com os dados
modelo_produtos.fit(dados_produtos)

# 3. Os centros dos clusters são os nossos produtos "âncora"
produtos_ancora = modelo_produtos.cluster_centers_

print(f"Características dos Produtos Âncora (Preço, Popularidade):\n{produtos_ancora}")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
