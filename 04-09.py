#Nesta MISSÃO 02 os alunos deverão desenvolver novos algorítmos de Machine Learning em Aprendizado Supervisionado e não supervisionado e um exercício comentando de aprendizdo por reforço.

#VALOR DESTA MISSÃO: 0,25 na AC


#EXEC-01 -  Aprendizado Supervisionado
#"No aprendizado supervisionado, nós damos ao modelo as perguntas (dados) e as respostas (rótulos) e ele precisa aprender a regra que os conecta. Pense em um gabarito de prova."

#Aprendizado Supervisionado (Classificação)
# -------------------------------------------------------------------------
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

print("--- Exercício 1 -  Missão 2 (Aprendizado Supervisionado) ---")

# Dados: [nota_prova_1, nota_trabalho_2]
# Rótulos: 0 = Reprovou, 1 = Passou
# Cloque notas no seu DataSet
X_treino = np.array([
    [1,7], [2,5], [8,9], [10,0], # Passou
    [9,5], [7,0], [3,2], [6,9]  # Reprovou
])
y_treino = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Criando o modelo. O KNN decide o rótulo de um novo ponto olhando para seus vizinhos mais próximos.
# n_neighbors=3 significa que ele vai consultar os 3 vizinhos mais próximos.
modelo_knn = KNeighborsClassifier(n_neighbors=3)

# O '.fit()' é onúcleo do aprendizado: o modelo analisa os dados e ajusta seus parâmetros.
modelo_knn.fit(X_treino, y_treino)

# testar com novos alunos.
aluno_A = np.array([[8,9]]) # Esperamos que passe (1)
aluno_B = np.array([[5,4]]) # Esperamos que reprove (0)

# O '.predict()' usa o conhecimento adquirido para fazer uma previsão.
previsao_A = modelo_knn.predict(aluno_A)
previsao_B = modelo_knn.predict(aluno_B)

print(f"Dados de treino (Notas): \n{X_treino}")
print(f"Rótulos de treino (Situação): {y_treino}")
print("-" * 20)
print(f"Previsão para o Aluno A: {'Passou' if previsao_A[0] == 1 else 'Reprovou'}")
print(f"Previsão para o Aluno B: {'Passou' if previsao_B[0] == 1 else 'Reprovou'}")
print("-" * 50, "\n")




#EXEC-02 - Aprendizado Supervisionado (Regressão)
#Crie um modelo que prevê o preço de um imóvel com base na sua área (m²) e no número de quartos. Usem LinearRegression."
# -----------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

print("--- Exercício 2 -  Missão 2 (Aprendizado Supervisionado) ---")

# Dados: [área_m2, numero_quartos]
# Rótulos: preco_em_milhares_de_reais
X_imoveis = np.array([
    [60, 2], [75, 3], [80, 3], # Imóveis menores
    [120, 3], [150, 4], [200, 4] # Imóveis maiores
])
y_precos = np.array([150, 200, 230, 310, 400, 500])

# TODO: Crie uma instância do modelo LinearRegression.
modelo_regressao = LinearRegression();

# TODO: Treine o modelo com os dados de imóveis (X_imoveis, y_precos).
modelo_regressao.fit(X_imoveis, y_precos)

# TODO: Crie um novo imóvel para testar (ex: 100m², 3 quartos).
imovel_teste = np.array([[16, 3]])

# TODO: Faça a previsão do preço para o novo imóvel.
preco_previsto = modelo_regressao.predict(imovel_teste)

print(f"Previsão de preço para um imóvel de 16m² com 3 quartos: R$ {preco_previsto[0]:.2f} mil")
#print("Complete o código acima!")
print("-" * 50, "\n")

#EXEC-03

import numpy as np
from sklearn.cluster import KMeans

print("--- Exercício 3 -  Missão 2 (Aprendizado Supervisionado) ---")

clientes = np.array([
    [30, 1], [45, 2], [35, 1], # Grupo de baixo valor/frequência
    [4000, 8], [500, 2], [550, 9] # Grupo de alto valor/frequência
])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

clusters_encontrados = kmeans.fit_predict(clientes)

print(f"Dados dos clientes (sem rótulos):\n{clientes}")
print(f"Clusters encontrados pelo KMeans para cada cliente: {clusters_encontrados}")
print("Observe como o algoritmo separou corretamente os clientes nos grupos 0 e 1.")
print("-" * 50, "\n")

#EXEC-04

import numpy as np
from sklearn.cluster import KMeans

print("--- Exercício 4 -  Missão 2 (Aprendizado Não Supervisionado) ---")

# Dados: [valor_transacao, hora_do_dia (0-23)]
transacoes = np.array([
    [15.50, 14], [30.00, 10], [12.75, 11],
    [50.20, 19], [20005.00, 9],
    [2500.00, 3] # Uma transação muito alta e de madrugada -> suspeita
])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
modelo_anomalia = kmeans

# TODO: Treine e preveja os clusters para os dados de transações.
clusters_transacoes = kmeans.fit_predict(transacoes)

print(f"Clusters para as transações: {clusters_transacoes}")
print("A transação anômala é aquela que está em um cluster isolado!")
print("Complete o código acima!")
print("-" * 50, "\n")

#EXEC-05

import time

POSICAO_INICIAL = 0
POSICAO_COMIDA = 10
recompensa_total = 0

# O agente começa na posição inicial.
posicao_agente = POSICAO_INICIAL

print("--- Iniciando a Simulação do PERSONAGEM COMILÃO ---")
print(f"O agente começa na posição {posicao_agente} e quer chegar na comida na posição {POSICAO_COMIDA}.")
print("-" * 30)

# O agente tem no máximo 10 passos para tentar chegar à comida.
for passo in range(10):
    print(f"Passo {passo + 1}:")
    posicao_agente += 1

    # O agente sempre toma a mesma AÇÃO: mover para a 'direita'.
    acao_agente = 'direita'
    print(f"   -> Ação do Agente: '{acao_agente}'")

    # ======================================================================
    # CONSTRUA A LÓGICA DO AMBIENTE
    # ======================================================================

    # Calcule a recompensa do passo
    recompensa_do_passo = -1
    if posicao_agente == POSICAO_COMIDA:
        recompensa_do_passo = +20

    # Atualize a recompensa total
    recompensa_total += recompensa_do_passo

    # Exibe o resultado do passo
    print(f"   <- Resposta do Ambiente: Nova Posição={posicao_agente}, Recompensa={recompensa_do_passo}")

    # Verifique se o jogo terminou
    if posicao_agente == POSICAO_COMIDA:
        print("chegou na comida")
        break

    # Pausa para visualização
    time.sleep(1)
    print("-" * 30)

print("-" * 30)
print(f"Simulação Finalizada! Recompensa total acumulada: {recompensa_total}")



