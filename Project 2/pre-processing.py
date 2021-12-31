import glob
import nltk
import copy
import pandas as pd
    
class preProcessing:
    def __init__(self,path):
        self.vocabulary = self.getVocab(path)
        
    def read_data(self,pos_path,neg_path):
        #get the path of all files
        file_names = [file_name for file_name in glob.glob(pos_path)]
        self.labels = ['pos' for i in range(len(file_names))]
        file_names += [file_name for file_name in glob.glob(neg_path)]
        self.labels += ['neg' for i in range(len(file_names)-len(self.labels))]
        # self.labels = self.labels[:10]
        # file_names = file_names[:10]
        self.corpus_set = []
        #read the text of all files
        for file_name in file_names:
            with open(file_name,encoding="utf8") as f:
                self.corpus_set.append(f.read())
            f.close()
    
    def tokenize(self, text_list):
        word_text_list = []
        for text in text_list:
            #normalize the case
            text = text.lower()
            #sperate the puntuations from word
            word_list = [word for word in nltk.word_tokenize(text)]
            word_text_list.append(word_list)
        return word_text_list
    
    def getVocab(self, file_name):
        vocab = {}
        #read the vocabulary file
        with open(file_name,encoding="utf8") as f:
            for word in f:
                vocab[word.strip()] = 0
        f.close()
        return vocab
    
    def word_to_vector(self,word_list,vocab):
        #covert the word list into bow vector
        vector_dict = copy.deepcopy(vocab)
        for word in word_list:
            if word in vocab:
                vector_dict[word] += 1
        return list(vector_dict.values())
    
    def get_matrix(self, word_text_list,vocab):
        # create the bow matrix
        bow_matrix = [self.word_to_vector(words,vocab) for words in word_text_list]
        return bow_matrix
    
    def list_to_string(self, bow_matrix):
        # convert the list of features into string 
        bow_string_matrix = []
        for i in range(len(bow_matrix)):
            bow_string_matrix.append(' '.join([str(num) for num in bow_matrix[i]]))
        return bow_string_matrix
        
        
    
    def write_bow_vector(self, bow_matrix,file_name):
        #write the bag of words vector
        df = pd.DataFrame()
        df['labels'] = self.labels
        df['features'] = bow_matrix
        df.to_csv(file_name+'.csv', index=False)
        
    def run_script(self,pos_path,neg_path,out_file):
        # print(path)
        vocab = self.vocabulary
        self.read_data(pos_path,neg_path)
        corpus_words_set = self.tokenize(self.corpus_set)
        bow_matrix = self.get_matrix(corpus_words_set,vocab)
        bow_string_matrix = self.list_to_string(bow_matrix)
        self.write_bow_vector(bow_string_matrix,out_file)

if __name__ == "__main__":
    obj = preProcessing('dataset\\imdb.vocab')
    obj.run_script('dataset\\train\\pos\\*','dataset\\train\\neg\\*','movies_training_features')
    obj.run_script('dataset\\test\\pos\\*','dataset\\test\\neg\\*','movies_testing_features')