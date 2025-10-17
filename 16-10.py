EXERCÍCIO 1



#Passo 1: Carregar e Pré-processar os Dados
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Verificar a forma dos dados
print(f"Forma dos dados de treinamento: {x_train.shape}")
print(f"Forma dos rótulos de treinamento: {y_train.shape}")
print(f"Forma dos dados de teste: {x_test.shape}")
print(f"Forma dos rótulos de teste: {y_test.shape}")

# Normalizar os valores dos pixels para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Achatar as imagens (28x28 -> 784)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"Forma após flatten: {x_train_flat.shape}")


#Passo 2: Construir a MLP
# Definir os nomes das classes para visualização
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Construir o modelo MLP
model = keras.Sequential([
    # Camada de entrada (flatten já foi feito manualmente)
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    # Camadas ocultas
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    
    # Camada de saída (10 classes)
    keras.layers.Dense(10, activation='softmax')
])

# Visualizar a arquitetura do modelo
model.summary()


#Passo 3: Compilar o Modelo
# Compilar o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


#Passo 4: Treinar o Modelo
# Definir callbacks para monitoramento
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Treinar o modelo
history = model.fit(
    x_train_flat, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test_flat, y_test),
    callbacks=[early_stopping],
    verbose=1
)


#Passo 5: Avaliar e Visualizar
# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")
print(f"Perda no conjunto de teste: {test_loss:.4f}")

# Plotar histórico de treinamento
plt.figure(figsize=(12, 4))

# Gráfico de acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

# Fazer previsões no conjunto de teste
predictions = model.predict(x_test_flat)
predicted_classes = np.argmax(predictions, axis=1)

# Visualizar algumas previsões
def plot_predictions(images, true_labels, predicted_labels, class_names, num_images=10):
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        
        # Cor do texto: verde se correto, vermelho se errado
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        
        plt.title(f'Verd: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}', 
                 color=color, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualizar as primeiras 10 imagens do teste
plot_predictions(x_test, y_test, predicted_classes, class_names)

# Matriz de confusão
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Calcular matriz de confusão
cm = confusion_matrix(y_test, predicted_classes)

# Plotar matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, predicted_classes, target_names=class_names))

# Analisar exemplos específicos
def analyze_predictions(true_labels, predicted_labels, probabilities, class_names, num_examples=5):
    # Encontrar exemplos corretos e incorretos
    correct_indices = np.where(true_labels == predicted_labels)[0]
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    
    print(f"\nExemplos de acertos:")
    for i in range(min(3, len(correct_indices))):
        idx = correct_indices[i]
        print(f"Imagem {idx}: {class_names[true_labels[idx]]} - "
              f"Confiança: {probabilities[idx][predicted_labels[idx]]:.4f}")
    
    print(f"\nExemplos de erros:")
    for i in range(min(3, len(incorrect_indices))):
        idx = incorrect_indices[i]
        print(f"Imagem {idx}: Verdadeiro: {class_names[true_labels[idx]]} - "
              f"Predito: {class_names[predicted_labels[idx]]} - "
              f"Confiança: {probabilities[idx][predicted_labels[idx]]:.4f}")

analyze_predictions(y_test, predicted_classes, predictions, class_names)



EXERCÍCIO 2

#Exercício 2: Reconhecimento de Dígitos com CNN (0,10pts)

#Passo 1: Carregar e Pré-processar os Dados

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Verificar a forma dos dados
print(f"Forma dos dados de treinamento: {x_train.shape}")
print(f"Forma dos rótulos de treinamento: {y_train.shape}")
print(f"Forma dos dados de teste: {x_test.shape}")
print(f"Forma dos rótulos de teste: {y_test.shape}")

# Normalizar os valores dos pixels para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionar para o formato esperado pelas camadas convolucionais (28, 28, 1)
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"Forma após redimensionamento - Treino: {x_train_cnn.shape}")
print(f"Forma após redimensionamento - Teste: {x_test_cnn.shape}")

# Visualizar algumas imagens do dataset
def plot_sample_images(images, labels, num_images=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_images(x_train_cnn, y_train)

#Passo 2: Construir a CNN
# Construir o modelo CNN
model_cnn = keras.Sequential([
    # Primeira camada convolucional
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Segunda camada convolucional
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Terceira camada convolucional
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Achatar a saída para camadas densas
    keras.layers.Flatten(),
    
    # Camadas densas
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Regularização para evitar overfitting
    
    # Camada de saída (10 classes - dígitos 0-9)
    keras.layers.Dense(10, activation='softmax')
])

# Visualizar a arquitetura do modelo
model_cnn.summary()

#Passo 3: Compilar e Treinar
# Compilar o modelo
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para melhor treinamento
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
]

# Treinar o modelo
history_cnn = model_cnn.fit(
    x_train_cnn, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_test_cnn, y_test),
    callbacks=callbacks,
    verbose=1
)

#Passo 4: Avaliar e Analisar
# Avaliar o modelo no conjunto de teste
test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\n=== RESULTADOS DA CNN ===")
print(f"Acurácia no conjunto de teste: {test_accuracy_cnn:.4f}")
print(f"Perda no conjunto de teste: {test_loss_cnn:.4f}")

# Plotar histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Gráfico de acurácia
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Treinamento', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    plt.title('Acurácia do Modelo CNN', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de perda
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Treinamento', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title('Perda do Modelo CNN', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico da taxa de aprendizado
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], linewidth=2, color='purple')
        plt.title('Taxa de Aprendizado', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_cnn)

# Fazer previsões
predictions_cnn = model_cnn.predict(x_test_cnn)
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)

# Matriz de confusão
def plot_confusion_matrix_cm(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão - CNN', fontsize=16)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

plot_confusion_matrix_cm(y_test, predicted_classes_cnn, range(10))

# Relatório de classificação detalhado
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, predicted_classes_cnn))

# Visualizar previsões corretas e incorretas
def visualize_predictions(images, true_labels, predicted_labels, probabilities, num_examples=10):
    # Encontrar exemplos corretos e incorretos
    correct_indices = np.where(true_labels == predicted_labels)[0]
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    
    print(f"Total de acertos: {len(correct_indices)}/{len(true_labels)}")
    print(f"Total de erros: {len(incorrect_indices)}/{len(true_labels)}")
    
    # Plotar alguns exemplos corretos
    plt.figure(figsize=(15, 6))
    plt.suptitle('Exemplos de Previsões Corretas', fontsize=16)
    
    for i in range(min(5, len(correct_indices))):
        idx = correct_indices[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        confidence = probabilities[idx][predicted_labels[idx]]
        plt.title(f'Verd: {true_labels[idx]}, Pred: {predicted_labels[idx]}\nConf: {confidence:.4f}', 
                 color='green', fontsize=10)
        plt.axis('off')
    
    # Plotar alguns exemplos incorretos
    for i in range(min(5, len(incorrect_indices))):
        idx = incorrect_indices[i]
        plt.subplot(2, 5, i + 6)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        confidence = probabilities[idx][predicted_labels[idx]]
        plt.title(f'Verd: {true_labels[idx]}, Pred: {predicted_labels[idx]}\nConf: {confidence:.4f}', 
                 color='red', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(x_test_cnn, y_test, predicted_classes_cnn, predictions_cnn)


EXERCÍCIO 3

# Passo 1: Carregar e Pré-processar os Dados
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Definir o vocabulário máximo e o comprimento máximo das sequências
max_words = 10000
maxlen = 250

# Carregar o dataset IMDB
# num_words=max_words mantém apenas as palavras mais frequentes
(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = imdb.load_data(num_words=max_words)

print(f"Número de sequências de treinamento: {len(x_train_imdb)}")
print(f"Número de sequências de teste: {len(x_test_imdb)}")

# Padronizar o comprimento das sequências
# sequências mais curtas são preenchidas com zeros, sequências mais longas são truncadas
x_train_padded = pad_sequences(x_train_imdb, maxlen=maxlen)
x_test_padded = pad_sequences(x_test_imdb, maxlen=maxlen)

print(f"Forma após padding - Treino: {x_train_padded.shape}")
print(f"Forma após padding - Teste: {x_test_padded.shape}")

# Visualizar um exemplo (sequência pré-processada)
print("\nExemplo de sequência pré-processada:")
print(x_train_padded[0])
print(f"Rótulo do exemplo: {y_train_imdb[0]}")

# Passo 2: Construir a Rede LSTM
# Definir as dimensões da camada de embedding
embedding_dim = 128

# Construir o modelo LSTM
model_lstm = keras.Sequential([
    # Camada de Embedding
    # input_dim: tamanho do vocabulário (max_words)
    # output_dim: dimensão dos vetores de embedding (embedding_dim)
    # input_length: comprimento das sequências de entrada (maxlen)
    keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen),

    # Camada LSTM
    # Units: número de unidades na camada LSTM
    keras.layers.LSTM(64),

    # Camada Densa de Saída
    # 1 unidade de saída para classificação binária (positivo/negativo)
    # Ativação 'sigmoid' para saída entre 0 e 1
    keras.layers.Dense(1, activation='sigmoid')
])

# Visualizar a arquitetura do modelo
model_lstm.summary()

# Passo 3: Compilar e Treinar

# Compilar o modelo
model_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Perda para classificação binária
    metrics=['accuracy']        # Métrica de avaliação
)

# Definir callbacks para monitoramento (opcional, mas recomendado)
callbacks_lstm = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,  # Parar o treinamento se a perda de validação não melhorar por 3 épocas
        restore_best_weights=True # Restaurar os pesos do modelo da melhor época
    )
]

# Treinar o modelo
# Usar os dados padronizados (x_train_padded, x_test_padded)
history_lstm = model_lstm.fit(
    x_train_padded, y_train_imdb,
    epochs=10,          # Número de épocas de treinamento
    batch_size=32,      # Tamanho do lote
    validation_data=(x_test_padded, y_test_imdb), # Dados para validação
    callbacks=callbacks_lstm, # Callbacks
    verbose=1           # Mostrar progresso do treinamento
)

# Passo 4: Avaliar e Testar

# Avaliar o modelo no conjunto de teste
print("\nAvaliando o modelo no conjunto de teste...")
loss, accuracy = model_lstm.evaluate(x_test_padded, y_test_imdb, verbose=0)

print(f"Perda no conjunto de teste: {loss:.4f}")
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")


# Testar com frases personalizadas
def predict_sentiment(review_text):
    # Pré-processar a frase de teste
    # 1. Tokenizar a frase (converter palavras em índices)
    #    Usamos o word_index do dataset IMDB. Palavras fora do vocabulário serão tratadas.
    word_index = imdb.get_word_index()
    # Adicionar 3 para compensar os índices reservados (padding, start, unknown)
    indexed_review = [word_index.get(word.lower(), 2) + 3 for word in review_text.split()] # 2 é o índice para 'unknown'

    # 2. Padronizar a sequência para o mesmo comprimento usado no treinamento
    padded_review = pad_sequences([indexed_review], maxlen=maxlen)

    # Fazer a previsão
    prediction = model_lstm.predict(padded_review)

    # Interpretar a previsão (sigmoid output está entre 0 e 1)
    # Um valor próximo de 1 indica sentimento positivo, próximo de 0 indica negativo
    sentiment = "Positivo" if prediction[0][0] > 0.5 else "Negativo"
    confidence = prediction[0][0] if sentiment == "Positivo" else (1 - prediction[0][0])

    print(f"\nFrase: '{review_text}'")
    print(f"Previsão de Sentimento: {sentiment} (Confiança: {confidence:.4f})")

# Exemplos de frases para testar
predict_sentiment("This movie was absolutely fantastic! I loved every moment.")
predict_sentiment("The plot was boring and the acting was terrible. A complete waste of time.")
predict_sentiment("It was an okay film, nothing special but not bad either.")
predict_sentiment("I highly recommend this film to everyone!")
predict_sentiment("Worst experience ever. I would not watch it again.")
