from sklearn.neural_network import MLPClassifier
import numpy
def softmax(set):
    id_predect=[]
    new_set=[numpy.exp(x)/numpy.sum(numpy.exp(x),axis=0) for x in set]
    for item in new_set:
        highest,sol=-100,0
        for iter,section in enumerate(item):
            (highest,sol)=(section,iter) if section>highest else (highest,sol)
        id_predect.append(sol)
    return id_predect

class NeuralNet:
    def __init__(self,Data,logis_num):
        self.training_ID=(Data[0],Data[2])
        self.test_ID=(Data[1],Data[3])
        self.function=softmax
        self.num_logis=logis_num
    def createAndTest_network(self):
        network=MLPClassifier(max_iter=10000)
        network.fit(self.training_ID[0],self.training_ID[1])
        predict_id=self.function(network.predict_proba(self.test_ID[0]))
        id_with_test=zip(self.test_ID[1],predict_id)
        micro,macro=self.micro_macro(id_with_test,6)
        print(micro/43,macro/self.num_logis)

    def micro_macro(self,set,number):
        logits,t_logits=[0]*number,[0]*number
        micro,macro=0,0

        for id,predict in set:
            t_logits[predict]=t_logits[predict]+1 if predict==id else t_logits[predict]
            micro=micro+1 if predict==id else micro
            logits[predict]+=1
        new_set=zip(logits,t_logits)
        macro=sum([t_log/log for log,t_log in new_set])
        return micro,macro