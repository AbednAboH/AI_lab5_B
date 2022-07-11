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
                 serviving_mechanizem, mutation, gene_dist, max_iter, check, mutation_probability=0):
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
        self.check_experts = check
        self.options = self.create_options()

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

    def individual_fitness(self, citizen):
        confusion_mat, _ = citizen.NeuNet.test_network()
        citizen.fitness = confusion_mat.sum() / confusion_mat.trace()
        return citizen.fitness

    def fitness(self):
        for index, pop in enumerate(self.population):
            self.population[index].fitness = self.individual_fitness(pop)

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
    def mutate2(self,member):
        ipos = random.randint(0, len(member) - 1)
        delta = random.randrange(0,1)
        member= member[:ipos] + [delta] + member[ipos + 1:]
        return member
    def flatten(self, coefs):

        arr = coefs[0]
        for i in range(1, len(coefs)):
            arr = numpy.array(arr).ravel()
            new = numpy.array(coefs[i]).ravel()
            arr = numpy.concatenate((arr, new), dtype="float")
        arr.ravel()
        return [float(x) for x in arr]

    def unflaaten(self, coefs, input):
        initial_shape = len(coefs)
        new_array = []
        start = 0
        for i in range(initial_shape):
            shape = numpy.shape(coefs[i])
            temp = numpy.array(input[start:start + shape[0] * shape[1]])

            temp = temp.reshape(shape[0], shape[1])
            new_array.append(temp)
            start += shape[0] * shape[1]
        return new_array

    def cross(self, esize, gen, population, birth_count, fitnesstype, mut_type, prob_spec):
        for i in range(esize, esize + birth_count):
            self.buffer[i] = prob_spec()
            citizen1 = prob_spec()
            citizen2 = prob_spec()
            # condition = True
            i1, i2 = self.selection_methods.method[self.selection](population, self.fitness_array)
            # counter+=1
            io1 = self.flatten(i1.NeuNet.network.coefs_)
            io2 = self.flatten(i2.NeuNet.network.coefs_)
            citizen1_neurons, citizen2_neurons = self.cross_func[
                self.cross_type if not fitnesstype else CROSS1](io1, io2)
            citizen1.create_object(0, 0)
            citizen2.create_object(0, 0)
            mutation = GA_MUTATION
            if random.randint(0, maxsize) < mutation:
                citizen1_neurons=self.mutate2(citizen1_neurons)
                citizen2_neurons=self.mutate2(citizen2_neurons)
            citizen1.NeuNet.network.coefs_ = self.unflaaten(i1.NeuNet.network.coefs_, citizen1_neurons)
            citizen2.NeuNet.network.coefs_ = self.unflaaten(i2.NeuNet.network.coefs_, citizen2_neurons)
            self.individual_fitness(citizen1)
            self.individual_fitness(citizen2)
            # print("after",citizen1.fitness,citizen2.fitness)
            winner=citizen1 if citizen1.fitness < citizen2.fitness else citizen2

            self.buffer[i] = winner if winner.fitness < self.population[i].fitness else self.population[i]

    def algo(self, i):
        self.fitness_array = self.propablities_rank_based(len(self.population) - 1, self.population)
        self.mate(i, 0, 0, self.prob_spec, self.population)  # mate the population together
        self.buffer, self.population = self.population, self.buffer
        self.population = self.sort_by_fitness(self.population)
        self.solution = self.population[0]

    def stopage(self, i):
        return i == self.iteration or self.solution.fitness == 1


linear_scale = lambda x: x[0] * x[1] + x[2]


# (s,mio,i)
def p_linear_rank(mio, i, s=1.5):
    if mio > 1:
        return (2 - s) / mio + 2 * i * (s - 1) / (mio * (mio - 1))
    else:
        return 1

# the general idea is to create 2 genetic algorithims one that works on a given population and given elite members
# the second one uses the first class to work on each spiecy and then adds the solutions together
