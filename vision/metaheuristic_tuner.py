"""
Optimizador de Hiperparámetros mediante Algoritmos Genéticos (Metaheurística).
Este script busca la mejor configuración para la CNN de DermaScan.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import copy

# Mock fitness - En un caso real esto entrenaría el modelo
def calculate_fitness(chromosome):
    """
    Evalúa un individuo (parámetros). 
    En producción, esto ejecutaría un entrenamiento corto y devolvería el Accuracy en validación.
    """
    lr = chromosome['lr']
    dropout = chromosome['dropout']
    weight_decay = chromosome['weight_decay']
    
    # Simulación de función de fitness (basada en el conocimiento de que 
    # lr=3e-4 y dropout=0.3 suelen ser buenos para este tipo de transfer learning)
    optimal_lr = 0.0003
    optimal_dropout = 0.3
    
    # El fitness es mayor cuanto más cerca esté de los "buenos" parámetros + un poco de aleatoriedad
    score = 100.0
    score -= abs(np.log10(lr) - np.log10(optimal_lr)) * 20
    score -= abs(dropout - optimal_dropout) * 30
    score += random.uniform(-2, 2) # Ruido
    
    return max(0, score)

class GeneticOptimizer:
    def __init__(self, population_size=10, generations=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.population = []

    def _create_individual(self):
        """Crea un individuo aleatorio (Cromosoma)."""
        return {
            'lr': 10**random.uniform(-5, -2), # 1e-5 a 1e-2
            'dropout': random.uniform(0.1, 0.5),
            'weight_decay': 10**random.uniform(-6, -3)
        }

    def _crossover(self, parent1, parent2):
        """Cruce por promedio o punto de corte."""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, individual):
        """Mutación aleatoria."""
        mutated = individual.copy()
        gene_to_mutate = random.choice(list(mutated.keys()))
        
        if gene_to_mutate == 'lr':
            mutated['lr'] *= random.choice([0.5, 2.0])
        elif gene_to_mutate == 'dropout':
            mutated['dropout'] = max(0.1, min(0.6, mutated['dropout'] + random.uniform(-0.1, 0.1)))
        
        return mutated

    def optimize(self):
        print(f"[Metaheurística] Iniciando búsqueda genética (Población: {self.population_size}, Gen: {self.generations})")
        
        # 1. Población inicial
        self.population = [self._create_individual() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            # 2. Evaluación
            scores = [(self.calculate_individual_fitness(ind), ind) for ind in self.population]
            scores.sort(key=lambda x: x[0], reverse=True)
            
            best_score, best_ind = scores[0]
            print(f"  G{gen}: Mejor Fitness = {best_score:.2f} | params: {best_ind}")
            
            # 3. Selección (Elitismo: mantenemos los 2 mejores)
            new_population = [scores[0][1], scores[1][1]]
            
            # 4. Reproducción
            while len(new_population) < self.population_size:
                p1 = random.choice(self.population[:5]) # Torneo/Selección de los mejores
                p2 = random.choice(self.population[:5])
                child = self._crossover(p1, p2)
                
                # 5. Mutación
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            self.population = new_population

        return scores[0][1]

    def calculate_individual_fitness(self, ind):
        # Aquí llamaríamos a la función de entrenamiento real
        return calculate_fitness(ind)

if __name__ == "__main__":
    tuner = GeneticOptimizer(population_size=12, generations=8)
    best_params = tuner.optimize()
    print("\n[FIN] Mejores parámetros encontrados por el Algoritmo Genético:")
    print(best_params)
