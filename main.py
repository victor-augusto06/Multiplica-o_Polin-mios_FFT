import numpy as np
import pyfftw
import time 
import random
import concurrent.futures

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

def execucao_sequencial(listaA , listaB):
    resultados = []
    for poly_a, poly_b in zip(listaA, listaB):
        resultados.append(multiplicar_fft(poly_a,poly_b))
    return resultados

def execucao_paralela(listaA, listaB, num_threads):
    resultados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        result = executor.map(multiplicar_fft, listaA, listaB)
        resultados = list(result)
    return resultados

print("Sistema iniciado!\n")
print("Você deve primeiro passar as instruções para o vetor a ser multiplicado seja criado!\n")

try:
    n=int(input("Qual deve ser o número do tamanho do vetor?\n"))
    grau_max=int(input("Qual deve ser o grau máximo dos vetores?\n"))

except ValueError:
    print("Entrada inválida. Por favor, insira apenas números inteiros.")
    exit()

print("\nGerando dados... Isso pode demorar um pouco.")
lista_polinomios_A, lista_polinomios_B = gerar_dados(n, grau_max)
print("Vetor criado com sucesso!")

while True:
    print("\n--- Menu de Execução ---")
    print("Escolha uma opção:")
    print("1. Executar versão Sequencial (procedural)")
    print("2. Executar versão Paralela (threads)")
    print("3. Sair")

    escolha = input("Digite sua opção (1, 2 ou 3): ")

    if escolha == '1':
        print("\nIniciando execução SEQUENCIAL...")
        inicio_seq = time.perf_counter()
        resultados_seq = execucao_sequencial(lista_polinomios_A, lista_polinomios_B)
        fim_seq = time.perf_counter()   
        tempo_seq = fim_seq - inicio_seq
        print(f"Execução SEQUENCIAL concluída.")
        print(f"Tempo total: {tempo_seq:.4f} segundos")
        print(f"Total de {len(resultados_seq)} multiplicações realizadas.")

    elif escolha == '2':
        try:
            qntd_threads = int(input("\nQuantas threads você quer utilizar? (ex: 8)\n"))
            if qntd_threads <= 0:
                 print("O número de threads deve ser maior que zero.")
                 continue
            print("\nIniciando execução PARALELA...")
            inicio_seq = time.perf_counter()
            resultados_seq = execucao_paralela(lista_polinomios_A, lista_polinomios_B, qntd_threads)
            fim_seq = time.perf_counter()   
            tempo_seq = fim_seq - inicio_seq
            print(f"Execução PARALELA concluída.")
            print(f"Tempo total: {tempo_seq:.4f} segundos")
            print(f"Total de {len(resultados_seq)} multiplicações realizadas.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")

    elif escolha == '3':
        print("\nSaindo do programa...")
        break
    else:
        print("\nOpção inválida. Por favor, digite 1, 2 ou 3.")