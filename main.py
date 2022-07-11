import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Genetic import C_genetic_algorithem
from settings import *
from neural_network import NeuralNet,new_ID

def read_data():
    CSV_file = pd.read_csv("inputs\glass.data")
    ID = LabelEncoder().fit_transform(CSV_file.values[:, -1])
    Norm_Data = pd.DataFrame(MinMaxScaler().fit_transform(CSV_file.values[:, 1:-1]))
    return ID, Norm_Data



def plot(iter, tag):

    plt.hist(iter,color="red",histtype="step",density=True,orientation="horizontal")
    plt.ylabel('fitness')
    plt.xlabel('iterations')
    plt.title(tag)
    CHECK_FOLDER = os.path.isdir(f"outputs\{tag}")
    if not CHECK_FOLDER:
        os.makedirs(f"outputs\{tag}")
    plt.savefig(f"outputs\{tag}\{tag}.png")
    plt.close()





#get array of stats against robots
def spacer(lines,f):
    d="-"*lines
    f.write(f"{d}\n")
def create_results_file(name,results,streak):
    f = open(fr"outputs\{name}\Results_{name}.txt", "a")
    spaceforfirst=15
    fitnes=23
    all_lines=fitnes*3+spaceforfirst
    for _ in range(all_lines-2):f.write("-")
    mid=23*7
    f.write("\n")
    for _ in range(mid):f.write(" ")
    f.write(f"|Results:|")
    f.write("\n")
    len_line=21
    space=23*15
    spacer(space, f)
    our_Robot="Our Agent's win Ratio"
    lenofR=len_line-len(our_Robot)
    f.write(our_Robot)
    for i in range(lenofR):f.write(" ")




    f.write("|")


    for robot in results[0]:
        len_space=len_line-len(robot[1])
        f.write(f" {robot[1]}")
        for i in range(len_space):f.write(" ")
        f.write("|")
    f.write("\n")

    spacer(space, f)

    for i in range(len_line):f.write(" ")
    f.write("|")
    for robot in results[0]:
        len_space=10-len("winer")
        num = len_space // 2
        num = num * 2 + len(robot[1])
        f.write(f" winer")
        for i in range(len_space):f.write(" ")
        f.write("|")
        len_space=10-len("score")
        f.write(f"score")
        for i in range(len_space):f.write(" ")
        f.write("|")
    f.write("\n")

    for row ,percentage in zip(results,streak):

        spacer(space, f)
        win_perc = int(percentage * 100)

        f.write(f"{win_perc}%")
        for i in range(len_line-len(str(win_perc))-1): f.write(" ")
        f.write("|")


        for item in row:
            (lenth,val)=(len("True"),"True") if item[0] else (len("False"),"False")
            len_space = 10 -lenth
            f.write(f" {item[0]}")
            for i in range(len_space): f.write(" ")
            f.write("|")
            len_space = 10 -len(str(item[2]))
            f.write(f"{item[2]}")
            for i in range(len_space): f.write(" ")
            f.write("|")

        f.write("\n")
    print(results)
    output=[[] for i in range(len(results[0]))]
    for i in range(len(results[0])):
        for row in results:
            output[i].append(row[i][0])

    spacer(space, f)
    averages = [int(100*sum(array) / len(array)) for array in output]
    lenofR = len_line - len(our_Robot)
    f.write(our_Robot)
    f.write("|")

    for i in range(lenofR): f.write(" ")
    for robot in results[0]:
        len_space=len_line-len(robot[1])
        f.write(f" {robot[1]}")
        for i in range(len_space):f.write(" ")
        f.write("|")
    f.write("\n")

    spacer(space, f)
    avrg_win_ratio = sum(streak) / len(streak)
    win_perc = int(avrg_win_ratio * 100)

    f.write(f"{win_perc}%")
    for i in range(len_line - len(str(win_perc)) - 1): f.write(" ")
    f.write("|")

    for average in averages:
        len_space=len_line-len(str(average))
        f.write(f" {str(average)}")
        for i in range(len_space):f.write(" ")
        f.write("|")
    f.write("\n")

def border():
    print("----------------------------------")



def main2():
    name=input("enter a name for the results file:\nthe results belong to the tournement between our agent against all other agents , not all participants against each other \n")
    target_size=1000
    # problem_set=RPS
    max_iter=int(input("enter number of max iterations !:"))
    GA_POPSIZE=int(input("enter population size:"))
    Check_experts=bool(int(input("check experts during co-evolution : \nnot recommended with high population size ,experts take too much time to think\ntrue: 1 \nfalse: 0")))
    # solution = C_genetic_algorithem(GA_TARGET, target_size, GA_POPSIZE, problem_set, problem_set, CX_, 0, 3,
    #                                 1, 1, 1, max_iter,Check_experts,mutation_probability=1)
    # output, iter, sol, output2, sol2, network, population=solution.solve()
def main():
    plot(new_ID,"distribution")
    nn=NeuralNet()
    micro,macro=nn.train_network()
    print(micro,macro)
if __name__ == "__main__":

    main()


