import pandas as pd
import sys
import pickle
import math

class NaiveBayes:
    def __init__(self):
        pass

    def read_files(self, filename):
        #read the files
        df = pd.read_csv(filename)
        #return the columns into list
        return df['features'].tolist(), df['labels'].tolist()
    
    def string_to_list(self,features):
        #convert the string into list
        bow_matrix = []
        for i in range(len(features)):
            bow_matrix.append([int(num) for num in features[i].split(' ')])
        return bow_matrix

    def prior_probability(self,labels, uni_class):
        #get the prior probaility of classes
        prior_prob = {uni_class[0]:len([class_ for class_ in labels if class_ == uni_class[0]])/len(labels),uni_class[1]:len([class_ for class_ in labels if class_ == uni_class[1]])/len(labels)}
        return prior_prob
    def vocab_count_per_class(self, bow_matrix,class_list,uni_class):
        #count the occurrence of each class
        occur_per_class = {uni_class[0]:0,uni_class[1]:0}
        for i in range(len(bow_matrix)):
            occur_per_class[class_list[i]] += sum(bow_matrix[i])
        return occur_per_class

    def get_prob_bow(self,occur_per_class,bow_matrix,class_list,uni_class):
        #count the probability of all classes
        prob_bow_dict = {uni_class[0]:[],uni_class[1]:[]}
        V = len(bow_matrix[0])
        for class_ in uni_class:
            for i in range(V):
                count = 0
                for j in range(len(bow_matrix)):
                    if class_ == class_list[j]:
                        count += bow_matrix[j][i]
                # add 1 smoothing
                prob = (count+1)/(occur_per_class[class_]+V)
                prob_bow_dict[class_].append(prob)
        return prob_bow_dict
    
    def save_model(self,prior_prob, prob_bow_dict, file_name):
        #save the model
        model_dic = {'Prior_Probability':prior_prob,'BOW_Probability':prob_bow_dict}
        with open(file_name, "wb") as write_file:
            pickle.dump(model_dic, write_file)
        write_file.close()
    
    def training_script(self,input_file, out_file):
        bow_matrix, labels = self.read_files(input_file)
        bow_matrix = self.string_to_list(bow_matrix)
        uni_class = list(set(labels))
        prior_prob = self.prior_probability(labels, uni_class)
        occur_per_class = self.vocab_count_per_class(bow_matrix,labels,uni_class)
        prob_bow_dict = self.get_prob_bow(occur_per_class,bow_matrix,labels,uni_class)
        self.save_model(prior_prob, prob_bow_dict, out_file)

    def load_model(self, file_name):
        #load the model
        with open(file_name,'rb') as read_file:
            model_dict = pickle.load(read_file)
        read_file.close()
        return model_dict.values()
    
    def get_predict_class_score(self,prob_bow_dict,prior_prob,bow_vector,uni_class):
        #get the score of both classes
        class_score = {uni_class[0]:prior_prob[uni_class[0]],uni_class[1]:prior_prob[uni_class[1]]}
        for class_ in uni_class:
            # prob = 1
            log_prob = 0
            for i in range(len(bow_vector)):
                if bow_vector[i] != 0:
                    # prob *= bow_vector[i]*prob_bow_dict[class_][i]
                    log_prob += math.log2(bow_vector[i]*prob_bow_dict[class_][i])
            # class_score[class_] = prob
            class_score[class_] = log_prob
        return class_score
    
    def getAccuracy(self,true_y, pred_y):
        #evalutaion
        count = 0
        for i in range(len(true_y)):
            if true_y[i] == pred_y[i]:
                count += 1
        return count/len(pred_y)

    
    def testing_script(self,input_file,model_file, out_file):
        bow_matrix, labels = self.read_files(input_file)
        bow_matrix = self.string_to_list(bow_matrix)
        prior_prob, prob_bow_dict = self.load_model(model_file)
        uni_class = list(prior_prob.keys())
        heading = 'predict_labels\n'
        predict_labels = []
        #save thhe file
        with open(out_file,'w') as write_file:
            write_file.write(heading)
            for i in range(len(labels)):
                classes_score = self.get_predict_class_score(prob_bow_dict,prior_prob,bow_matrix[i],uni_class)
                if classes_score[uni_class[0]] > classes_score[uni_class[1]]:
                    predict_labels.append(uni_class[0])
                    write_file.write("{}\n".format(uni_class[0]))
                else:
                    predict_labels.append(uni_class[1])
                    write_file.write("{}\n".format(uni_class[1]))
            write_file.write("Test Accuracy: {}\n".format(self.getAccuracy(labels,predict_labels)))
        write_file.close()

if __name__ == "__main__":
    training_file_name = sys.argv[1]
    testing_file_name = sys.argv[2]
    model_file_name = sys.argv[3]
    output_file_name = sys.argv[4]
    obj = NaiveBayes()
    obj.training_script(training_file_name,model_file_name)
    obj.testing_script(testing_file_name,model_file_name,output_file_name)

