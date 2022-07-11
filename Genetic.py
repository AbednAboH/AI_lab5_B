import random

from function_selection import cross_types
from algorithems import algortithem
from settings import *
from sys import maxsize
import numpy
import math


""" to generalize the problem we created a class that given a population calculates the solution"""


def adaptive_decrease(p, rate, generation):
    return 2 * (p ** 2) * math.exp(rate * generation) / (p + p * math.exp(rate * generation))


class C_genetic_algorithem(algortithem):
    Distance_hash = {}

    def __init__(self, target, tar_size, pop_size, problem_spec, problem_spec2, crosstype, fitnesstype, selection,
                 serviving_mechanizem, mutation, gene_dist, max_iter,check, mutation_probability=0):
        algortithem.__init__(self, target, tar_size, pop_size, problem_spec, fitnesstype, selection, max_iter)
        self.cross_func = cross_types().select
        self.cross_type = crosstype
        self.serviving = serviving_mechanizem
        self.mutation_type = mutation
        self.set_fitness = self.prob_spec().fitnesstype[gene_dist]
        self.fitness_func = self.prob_spec().fitnesstype
        self.selection_pressure = self.pop_diversity = 0
        self.hyper_mutaion = mutation_probability
        self.trigger_mutation = False
        self.elite_rate = GA_ELITRATE
        self.problem_spec2 = problem_spec2
        self.check_experts=check
        self.options = self.create_options()
        temp = problem_spec2()
        temp.create_object(self.target_size, self.target, self.options)
        # todo: add options to send to problem creation , we don't have to create it over and over again'

    def create_options(self):
        options = []
        for k in range(1, self.target_size + 1):
            for j in range(k + 1, self.target_size + 1):
                options.append([k, j])
        return options

    def init_population(self):
        super(C_genetic_algorithem, self).init_population()
        self.fitness()
        self.population = self.sort_by_fitness(self.population)
        self.solution = self.population[0]
        self.output.append(self.solution.fitness)
        self.iter.append(0)
        print(self.solution.fitness)

    def fitness(self):
        pass
    def propablities_rank_based(self, pop_size, population):

        # depending on the selection scheme get propabilities from ranking system !
        multiplier = 1 if self.selection == SUS or RWS else 0.5  # for now keep it like this
        # scale fitness values with linear ranking for sus and RWS
        fitness_array = []
        if self.selection == SUS:

            mio = pop_size
            fitness_array = numpy.array([p_linear_rank(mio, int(i)) for i in range(pop_size - 1, -1, -1)])
        # get accumulative sum of above values
        elif self.selection == RWS:
            mean = numpy.mean(fitness_array)
            std = numpy.std(fitness_array)
            # linear scale
            fitness_array = numpy.array([i for i in range((pop_size + 1) * 10, 10, -10)])
            # sigma scale
            fitness_array = numpy.array(
                [max((f - mean) / (2 * std), 1) if std > 0 else 1 for f in fitness_array])
            sum = fitness_array.sum()
            fitness_array = numpy.array([i / sum for i in fitness_array])

        else:
            # fps  for tournament selection
            fitness_array = numpy.array([pop.fitness for pop in population])
            sumof_fit = fitness_array.sum()
            sumof_fit = sumof_fit if sumof_fit else 1
            fitness_array = numpy.array([(pop.fitness + 1) / sumof_fit for pop in population])

        return fitness_array

    def mutate(self, member, type, fitnessType):
        member.mutate(self.target_size, member, type, self.options)

    def age_based(self, population):
        age_based_population = [citizen for citizen in population if 2 <= citizen.age <= 20]
        self.buffer[:len(age_based_population)] = age_based_population[:]
        return len(age_based_population)

    # selection of the servivng mechanizem
    def serviving_genes(self, gen, population):
        esize = math.floor(len(population) * self.elite_rate)
        if self.serviving == ELITIZEM:
            self.buffer[:esize] = population[:esize]
        # age
        elif self.serviving == AGE:
            esize = self.age_based(population)
        return esize

    # this function returns an array for each spiecy , how many are elite
    # so that we can choose the appropriate ammount of genes from speciy and so that the population size stayes the same

    def mate(self, gen, fitnesstype, mut_type, prob_spec, population):
        esize = self.serviving_genes(gen, population)
        self.cross(esize, gen, population, len(population) - esize, fitnesstype, mut_type, prob_spec)

    def cross(self, esize, gen, population, birth_count, fitnesstype, mut_type, prob_spec):

    def algo(self, i):
        self.fitness_array = self.propablities_rank_based(len(self.population) - 1, self.population)
        self.mate(i, 0, 0, self.prob_spec, self.population)  # mate the population together
        self.population = self.sort_by_fitness(self.population)
        self.solution = self.population[0]
    def stopage(self,i):
        return i==self.iteration or self.solution.fitness==1


linear_scale = lambda x: x[0] * x[1] + x[2]


# (s,mio,i)
def p_linear_rank(mio, i, s=1.5):
    if mio > 1:
        return (2 - s) / mio + 2 * i * (s - 1) / (mio * (mio - 1))
    else:
        return 1

# the general idea is to create 2 genetic algorithims one that works on a given population and given elite members
# the second one uses the first class to work on each spiecy and then adds the solutions together
