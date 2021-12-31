
import numpy as np

class Model:
    def __init__(self):
        self.train_corpus = []
        self.test_corpus = []
        self.unigram_dist = {}
        self.update_unigram_dist = {}
        self.bigram_dist = {}
        
    def readFiles(self):
        #read the training file
        with open('train.txt',encoding="utf8") as readFile:
            for text in readFile:
                self.train_corpus.append(text.strip())
        #read the testing file
        with open('test.txt',encoding="utf8") as readFile:
            for text in readFile:
                self.test_corpus.append(text.strip())
    
    def unigramDist(self):
        #Unigram Model
        for text in self.train_corpus:
            text = text.lower() + ' </s>'
            for word in text.split():
                if word in self.unigram_dist:
                    self.unigram_dist[word] += 1
                else:
                    self.unigram_dist[word] = 1
    def printWordTypes(self):
        count = 0
        unk_flag = False
        for word in self.unigram_dist:
            if self.unigram_dist[word] == 1:
                unk_flag = True
            else:
                count += 1
        if unk_flag:
            count += 1
        print('Word Types(Unique words) in training Corpus: {}'.format(count))

    def printWordTokens(self):
        count = 0
        for word in self.unigram_dist:
            count += self.unigram_dist[word]
        print('Word Tokens in training Corpus: {}'.format(count))

    def testWordTypesAndTokens(self):
        words_in_training = []
        words_not_in_training = []
        for text in self.test_corpus:
            text = text.lower() + ' </s>'
            for word in text.split():
                if word in self.unigram_dist:
                    words_in_training.append(word)
                else:
                    words_not_in_training.append(word)
        print('No I did not mapped the unknown words to <unk> in training and testing data')
        print('Word types percentage did not occur in training: {} %'.format((1-(len(set(words_in_training))/(len(set(words_in_training))+len(set(words_not_in_training)))))*100))
        print('Word tokens percentage did not occur in training: {} %'.format((1-(len(words_in_training)/(len(words_in_training)+len(words_not_in_training))))*100))

    def singletonToUnknown(self):
        for word in self.unigram_dist:
            if self.unigram_dist[word] == 1:
                if '<unk>' not in self.update_unigram_dist:
                    self.update_unigram_dist['<unk>'] = 0
                else:
                    self.update_unigram_dist['<unk>'] += 1
            else:
                self.update_unigram_dist[word] = self.unigram_dist[word]
    
    def bigramDist(self):
        for text in self.train_corpus:
            text = text.lower() + ' </s>'
            tokens = text.split()
            for i in range(1,len(tokens)):
                w0 = tokens[i-1]
                w1 = tokens[i]
                if w0 not in self.update_unigram_dist:
                    w0 = '<unk>'
                if w1 not in self.update_unigram_dist:
                    w1 = '<unk>'
                if (w0,w1) in self.bigram_dist:
                    self.bigram_dist[(w0,w1)] += 1
                else:
                    self.bigram_dist[(w0,w1)] = 1
    def preProcessing(self,text_corpus):
        text_list = []
        for text in text_corpus:
            text = text.lower()
            tokens = text.split()
            token_list = []
            for token in tokens:
                if token not in self.update_unigram_dist:
                    token_list.append('<unk>')
                else:
                    token_list.append(token)
            text_list.append('<s> '+' '.join(token_list)+" </s>")
        return text_list
    
    def savePreProcessingDataset(self):
        with open('pre_train.txt','w') as f:
            for text in self.preProcessing(self.train_corpus):
                f.write(text+'\n')
        with open('pre_test.txt','w') as f:
            for text in self.preProcessing(self.test_corpus):
                f.write(text+'\n')
        

    def testBigramTypesAndTokens(self):
        bigrams_in_training = []
        bigrams_not_in_training = []
        for text in self.test_corpus:
            text = text.lower() + ' </s>'
            tokens = text.split()
            for i in range(1,len(tokens)):
                w0 = tokens[i-1]
                w1 = tokens[i]
                if w0 not in self.update_unigram_dist:
                    w0 = '<unk>'
                if w1 not in self.update_unigram_dist:
                    w1 = '<unk>'
                if (w0,w1) in self.bigram_dist:
                    bigrams_in_training.append((w0,w1))
                else:
                    bigrams_not_in_training.append((w0,w1))
        print('Bigrams types percentage did not occur in training: {} %'.format((1-(len(set(bigrams_in_training))/(len(set(bigrams_in_training))+len(set(bigrams_not_in_training)))))*100))
        print('Bigrams tokens percentage did not occur in training: {} %'.format((1-(len(bigrams_in_training)/(len(bigrams_in_training)+len(bigrams_not_in_training))))*100))


    def unigramLogProb(self,text):
        unigramN = 0
        for word in self.update_unigram_dist:
            unigramN += self.update_unigram_dist[word]
        tokens = (text.lower()+' </s>').split()
        log_value = 0
        token_list = []
        for token in tokens:
            if token not in self.update_unigram_dist:
                token = '<unk>'
            prob = self.update_unigram_dist[token]/unigramN
            value = np.log2(prob)
            token_list.append((token,prob))
            log_value += value
        return log_value, token_list

    def bigramLogProb(self,text):
        tokens = (text.lower()+' </s>').split()
        log_value = 0
        token_list = []
        for i in range(1,len(tokens)):
            w0 = tokens[i-1]
            w1 = tokens[i]
            if w0 not in self.update_unigram_dist:
                w0 = '<unk>'
            if w1 not in self.update_unigram_dist:
                w1 = '<unk>'
            if (w0,w1) not in self.bigram_dist:
                bigram_occur = 0
            else:
                bigram_occur = self.bigram_dist[(w0,w1)]
            prob = bigram_occur/self.update_unigram_dist[w0]
            value = np.log2(prob)
            token_list.append(((tokens[i-1],tokens[i]),prob))
            log_value += value
        return log_value, token_list
    
    def bigramLogProbAddSmooth(self,text):
        tokens = (text.lower()+' </s>').split()
        log_value = 0
        token_list = []
        for i in range(1,len(tokens)):
            w0 = tokens[i-1]
            w1 = tokens[i]
            if w0 not in self.update_unigram_dist:
                w0 = '<unk>'
            if w1 not in self.update_unigram_dist:
                w1 = '<unk>'
            if (w0,w1) not in self.bigram_dist:
                bigram_occur = 0
            else:
                bigram_occur = self.bigram_dist[(w0,w1)]
            prob = (bigram_occur+1)/(self.update_unigram_dist[w0]+len(self.update_unigram_dist))
            value = np.log2(prob)
            token_list.append(((tokens[i-1],tokens[i]),prob))
            log_value += value
        return log_value, token_list
    
    def printSentenceLogProb(self):
        text = "I look forward to hearing your reply"
        log_prob, token_list = self.unigramLogProb(text)
        print('Unigram maximum likelihood Model')
        for token in token_list:
            print(token)
        print('\nLog probability: {}'.format(log_prob))
        print('Bigram maximum likelihood Model')
        log_prob, token_list = self.bigramLogProb(text)
        for token in token_list:
            print(token)
        print('\nLog probability: {}'.format(log_prob))
        print('Bigram Model with add one smoothing')
        log_prob, token_list = self.bigramLogProbAddSmooth(text)
        for token in token_list:
            print(token)
        print('\nLog probability: {}'.format(log_prob))
    
    def unigramPerplexity(self, corpus):
        sum = 0
        N = 0
        for text in corpus:
            words = text.split()
            N = N + len(words)+1 # +1 </s> tags
            sum = sum + self.unigramLogProb(text)[0]
        pp = np.power(2, -(sum/N))
        return pp
    
    def bigramPerplexity(self, corpus):
        sum = 0
        N = 0
        for text in corpus:
            words = text.split()
            N = N + len(words)+1 # +1 </s> tags
            sum = sum + self.bigramLogProb(text)[0]
        pp = np.power(2, -(sum/N))
        return pp
    
    def bigramPerplexityAddSmooth(self, corpus):
        sum = 0
        N = 0
        for text in corpus:
            words = text.split()
            N = N + len(words)+1 # +1 </s> tags
            sum = sum + self.bigramLogProbAddSmooth(text)[0]
        pp = np.power(2, -(sum/N))
        return pp
    
    def perplexitySenetnce(self):
        print('\n Perplexity of Sentence')
        text = "I look forward to hearing your reply"
        print('Unigram maximum likelihood Model')
        print(self.unigramPerplexity([text]))
        print('Bigram maximum likelihood Model')
        print(self.bigramPerplexity([text]))
        print('Bigram Model with add one smoothing')
        print(self.bigramPerplexityAddSmooth([text]))
    
    def perplexityTest(self):
        print('\n Perplexity of Test Set')
        print('Unigram maximum likelihood Model')
        print(self.unigramPerplexity(self.test_corpus))
        print('Bigram maximum likelihood Model')
        print(self.bigramPerplexity(self.test_corpus))
        print('Bigram Model with add one smoothing')
        print(self.bigramPerplexityAddSmooth(self.test_corpus))

if __name__ == "__main__":
    obj = Model()
    obj.readFiles()
    obj.unigramDist()
    obj.printWordTypes()
    obj.printWordTokens()
    obj.testWordTypesAndTokens()
    obj.singletonToUnknown()
    obj.bigramDist()
    obj.savePreProcessingDataset()
    obj.testBigramTypesAndTokens()
    obj.printSentenceLogProb()
    obj.perplexitySenetnce()
    obj.perplexityTest()

    

    
        


    

    