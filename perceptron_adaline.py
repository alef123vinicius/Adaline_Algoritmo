# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:14:04 2017

@author: alef1

perceptron adaline
trabalho

exemplo de entrada

teta     entradas         esperado
 1     0,0,...,n(25)         y         == total vetor[27] ... 
"""
import numpy as np

def main():
    pesos = np.zeros(26)
    entradas = [[-1, 1,-1,-1,-1, 1 , 1,-1,-1,-1, 1, -1,1,-1,1,-1, -1, 1,-1, 1,-1, -1,-1, 1,-1,-1, -1], #v
                [-1,-1,-1, 1,-1,-1 ,-1, 1,-1, 1,-1, -1,1,-1,1,-1,  1,-1,-1,-1, 1,  1,-1,-1,-1, 1,  1],  #v invertido
                [-1, 1,-1,-1,-1,1 ,1,-1,-1,-1,1, -1,1,-1,1,-1, -1,1,-1,1,-1, -1,1,1,1,-1, -1], #v
                [-1, -1,1,1,1,-1 ,-1,1,-1,1,-1, -1,1,-1,1,-1, 1,-1,-1,-1,1, 1,-1,-1,-1,1,  1],  #v invertido
                [-1, 1,-1,-1,-1,1 ,1,-1,-1,-1,1, -1,1,-1,1,-1, -1,-1,1,-1,-1, -1,-1,-1,-1,-1, -1], #v
                [-1, -1,-1,-1,-1,-1 ,-1,-1,1,-1,-1, -1,1,-1,1,-1, 1,-1,-1,-1,1, 1,-1,-1,-1,1,  1], #v invertido
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, -1], #v
                [-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,  1], #v invertido
                [-1,  1,-1,-1,-1,1,  -1,1,-1,-1,1,  -1,-1,1,-1,1,  -1,-1,-1,1,1,  -1,-1,-1,-1,1, -1]]#v
                
    # teste com a matriz restante representando mais tres rotulos de cada classe
    testes = [[-1, -1,-1,-1,-1,1, -1,-1,-1,-1,1, -1,-1,-1,-1,1, -1,-1,-1,-1,1,  1, 1, 1, 1, 1, -1], #v
              [-1,  1, 1, 1, 1,1, -1,-1,-1,-1,1, -1,-1,-1,-1,1, -1,-1,-1,-1,1, -1,-1,-1,-1, 1,  1], #v invertido
              [-1,  1,-1,-1,-1,1,  1,-1,-1,1,-1,  1,-1,1,-1,-1,  1,1,-1,-1,-1,  1,-1,-1,-1,-1, -1], #v
              [-1,  1,-1,-1,-1,-1, 1,1,-1,-1,-1,  1,-1,1,-1,-1,  1,-1,-1,1,-1,  1,-1,-1,-1, 1,  1], #v invertido
              [-1,  -1,-1,-1,-1,1, -1,-1,-1,1,1,  -1,-1,1,-1,1,  -1,1,-1,-1,1,  1,-1,-1,-1, 1,  1]] #v invertido
    treinamento(pesos, entradas)
    teste(pesos, testes)
        
def treinamento(pesos, entradas) :
    for epoca in range(3):
        for i in range(9):
            net = somatorio(pesos, entradas[i])
            y = f(net)
            erro = entradas[i][26] -  y
            if(erro):
                delta(pesos, 0.5, erro, entradas[i])
            print("amostra ",i," erro = ",erro," y = ",y," esperado = ", entradas[i][26])
        print("epoca = ",epoca+1,"\n")

def teste(pesos, testes):
     for i in range(5):
         net = somatorio(pesos, testes[i])
         y = f(net)
         print("Classificação do teste",i," y = ",y," esperado era = ",testes[i][26])
    
        
def delta(pesos, aprendizado, erro, entradas):
    for i in range(len(pesos)):
            pesos[i] += aprendizado*erro*entradas[i]
        
def somatorio(pesos, entradas):
    soma = 0;
    for i in range(len(pesos)):
        soma += pesos[i]*entradas[i]
    return soma

def f(net):
    if(net > 0):
        return 1
    return -1


if __name__ == "__main__":
    main()

