import numpy as np
import pyfftw
import time 
import random

def gerar_dados(N, grau_max):
    lista_A=[]
    lista_B=[]
    for i in range(N):
        grau_a = np.random.randint(1, grau_max + 1)
        grau_b = np.random.randint(1, grau_max + 1)
        lista_A.append(np.random.rand(grau_a + 1))
        lista_B.append(np.random.rand(grau_b + 1))
    return lista_A, lista_B

def multiplicar_fft(poly1, poly2):
    grau_dois = 0
    k_minimo = len(poly1)+len(poly2)-1
    while(k_minimo>2**grau_dois):
        grau_dois+=1
    k = 2**grau_dois

    p1_padded = pyfftw.empty_aligned(k,dtype='complex128')
    p2_padded = pyfftw.empty_aligned(k,dtype='complex128')
    p1_padded[:] = 0
    p2_padded[:] = 0
    p1_padded[:len(poly1)] = poly1
    p2_padded[:len(poly2)] = poly2

    fft_p1 = pyfftw.interfaces.numpy_fft.fft(p1_padded)
    fft_p2 = pyfftw.interfaces.numpy_fft.fft(p2_padded)
    
    fft_resultado = fft_p1 * fft_p2
    
    coeficientes_complexos = pyfftw.interfaces.numpy_fft.ifft(fft_resultado)

    coeficientes_reais = coeficientes_complexos.real
    coeficientes_arredondados = np.round(coeficientes_reais)
    
    arrayFinal = coeficientes_arredondados[:k_minimo]
    
    return arrayFinal

print("Sistema iniciado!\n")
print("Você deve primeiro passar as instruções para o vetor a ser multiplicado seja criado!\n")
n=int(input("Qual deve ser o número do tamanho do vetor?\n"))
grau_max=int(input("Qual deve ser o grau máximo dos vetores?"))
lista_polinomios_A, lista_polinomios_B = gerar_dados(n, grau_max)
print("Vetor criado com sucesso!")