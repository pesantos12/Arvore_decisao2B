import numpy as np
from collections import Counter
import math

# ================================================
# Estrutura do No (fornecida)

class No:
    def __init__(self, atributo=None, corte=None, esquerda=None, direita=None, classe=None):
        self.atributo = atributo    # indice do atributo (j)
        self.corte = corte          # valor do corte (s)
        self.esquerda = esquerda    # subarvore: x_j <= s
        self.direita = direita      # subarvore: x_j > s
        self.classe = classe        # classe da folha (se for no terminal)

# ================================================
# Função de impureza: Indice de Gini

def gini(y):
    if len(y) == 0:
        return 0.0
    contagem = Counter(y)
    proporcoes = [count / len(y) for count in contagem.values()]
    return 1.0 - sum(p ** 2 for p in proporcoes)

# ================================================
# Função de impureza: Entropia

def entropia(y):
    if len(y) == 0:
        return 0.0
    contagem = Counter(y)
    proporcoes = [count / len(y) for count in contagem.values()]
    return - sum(p * math.log2(p) for p in proporcoes)

# ================================================
# Classe majoritario (folha)
def classe_majoritaria(y):
    return Counter(y).most_common(1)[0][0]

# ================================================
# Critério de parada
def criterio_parada(dados, profundidade, max_profundidade):
    X, y = dados
    # No puro: todas as classes iguais
    if len(set(y)) <= 1:
        return True
    # Profundidade maxima
    if max_profundidade is not None and profundidade >= max_profundidade:
        return True
    # Poucas amostras (opcional)
    if len(y) < 2:
        return True
    return False

# ================================================
# Divisão dos dados
def dividir_dados(dados, j, s):
    X, y = dados
    mask = X[:, j] <= s
    dados_esq = (X[mask], y[mask])
    dados_dir = (X[~mask], y[~mask])
    return dados_esq, dados_dir

# ================================================
# Impureza media ponderada
def impureza_media_ponderada(dados_esq, dados_dir):
    X_esq, y_esq = dados_esq
    X_dir, y_dir = dados_dir
    n = len(y_esq) + len(y_dir)
    if n == 0:
        return 0.0
    i_esq = entropia(y_esq) if len(y_esq) > 0 else 0.0
    i_dir = entropia(y_dir) if len(y_dir) > 0 else 0.0
    return (len(y_esq) / n) * i_esq + (len(y_dir) / n) * i_dir

# ================================================
# Encontrar melhor divisao (j*, s*)
def melhor_divisao(dados):
    X, y = dados
    n_amostras, n_atributos = X.shape
    melhor_j, melhor_s = None, None
    melhor_impureza = float('inf')
    melhor_no_puro_tamanho = -1
    melhor_s_maior = -1

    for j in range(n_atributos):
        valores_unicos = np.unique(X[:, j])
        for s in valores_unicos:
            dados_esq, dados_dir = dividir_dados(dados, j, s)
            y_esq, y_dir = dados_esq[1], dados_dir[1]
            if len(y_esq) == 0 or len(y_dir) == 0:
                continue
            imp = impureza_media_ponderada(dados_esq, dados_dir)
            puro_esq = len(set(y_esq)) == 1 if len(y_esq) > 0 else False
            puro_dir = len(set(y_dir)) == 1 if len(y_dir) > 0 else False
            tamanho_puro = len(y_esq) if puro_esq else (len(y_dir) if puro_dir else 0)

            melhorar = False
            if imp < melhor_impureza:
                melhorar = True
            elif imp == melhor_impureza:
                if tamanho_puro > melhor_no_puro_tamanho:
                    melhorar = True
                elif tamanho_puro == melhor_no_puro_tamanho:
                    if s > melhor_s_maior:
                        melhorar = True
            if melhorar:
                melhor_impureza = imp
                melhor_j, melhor_s = j, s
                melhor_no_puro_tamanho = tamanho_puro
                melhor_s_maior = s
    if melhor_j is None:
        raise ValueError("Nenhuma divisão valida encontrada.")
    return melhor_j, melhor_s

# ================================================
# Função recursiva para construir a arvore (fornecida)
def construir_arvore(dados, profundidade=0, max_profundidade=None):
    X, y = dados

    if criterio_parada(dados, profundidade, max_profundidade):
        classe_folha = classe_majoritaria(y)
        return No(classe=classe_folha)
    
    # Escolhe a melhor divisao (atributo j, valor s)
    j, s = melhor_divisao(dados)
    dados_esq, dados_dir = dividir_dados(dados, j, s)

    # Criar o no atual e constroi as subarvores recursivamente
    no = No(atributo=j, corte=s)

    no.esquerda = construir_arvore(dados_esq, profundidade + 1, max_profundidade)
    no.direita = construir_arvore(dados_dir, profundidade + 1, max_profundidade)

    return no

# ================================================
# Função para imprimir a arvore (visualização didatica)
def imprimir_arvore(no, indent=""):
    if no.classe is not None:
        print(f"{indent}--> Prediz: Classe {no.classe}")
        return
    
    print(f"{indent}[X_{no.atributo + 1} <= {no.corte}]")
    print(f"{indent}|- Sim ", end ="")
    imprimir_arvore(no.esquerda, indent + "| ")
    print(f"{indent}|_ Não ", end ="")
    imprimir_arvore(no.direita, indent = "  ")

# ================================================
# Função para definir uma classe a partir de uma unidade de amostra

def definir_classe(no, valor):
    if no.classe is not None:
        print(f"Classe do valor {valor}: {no.classe}")
        return
    
    x1, x2 = valor
    if no.atributo == 0:
        corte = x1
    else:
        corte = x2

    if corte <= no.corte:
        definir_classe(no.esquerda, valor)
    else:
        definir_classe(no.direita, valor)


# ================================================
# Dados do exemplo (exatamento como no texto)
X = np.array([
    [1, 2], # 1
    [2, 1], # 2
    [3, 4], # 3
    [3, 6], # 4
    [4, 5], # 5
    [5, 3], # 6
    [6, 5], # 7
    [4, 1], # 8
])

y = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B', 'A'])

dados = (X, y)

# ================================================
# Construir e exibir a arvore
# ================================================
#print("Construindo arvore de decisão (CART com Entropia)...\n")
arvore = construir_arvore(dados, max_profundidade=None)
#print("Arvore de decisão construida:")
#imprimir_arvore(arvore)

definir_classe(arvore, [5, 4])