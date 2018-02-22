#import scholar.scholar as sch
from scipy import spatial
import numpy as np
import pickle as pkl

### Usage from other files ###
# import utils
# v = utils.vecMaster()
# word_list = v.expand(source_words, expansion_method, epsilon)

#def create_vector_object(sourcefile="data/fasttext.wiki.en.vec", destfile="data/fasttext.en", truncate=None):

### NOTE: run this function using python3
# otherwise the dict file is unusable
def create_fasttext_pkl(sourcefile="fasttexttrunc", destfile="data/fasttext.en", truncate=None):
    f=open(sourcefile,"r")
    firstline = f.readline()
    #vector_dict={}
    #token_dict={}
    tokens = []
    vectors = []
    for line in f:
        token = line[:line.index(' ')]
        vector_string = line[line.index(' ')+1:]
        vector = np.fromstring(vector_string, sep=' ')
        #token_dict[vector_string] = token
        #vector_dict[token] = vector
        tokens.append(token)
        #tokens.append(unitcode(token.decode('utf-8',errors='ignore')))
        vectors.append(vector)
    vectors = np.vstack(vectors)
    data = {}
    data['tokens'] = tokens
    data['vectors']= vectors
    f = open(destfile+'.pkl','wb')
    pkl.dump(data, f, protocol=4)
    f.close()

class vecMaster():

    def __init__(self,sourcefile='data/fasttext.en.pkl'):
        #self.scholar = sch.Scholar()
        with open(sourcefile,'rb') as myfile:
            data = pkl.load(myfile)
        self.tokens = data['tokens']
        self.vectors = data['vectors']

    def encode(self,word):
        pass

    def decode(self,word):
        pass

    def strip_tag(self,word):
        if '_' in word:
            return word[:word.index('_')]

    def expand(self,source_words, expansion_method='nearest_neighbor', epsilon=0.35):
        word_list = []
        for source_word in source_words:
            #try:
            #    tag = self.scholar.get_most_common_tag(source_word)
            #    print "Tag is " + tag
            #    source_vector = self.scholar.model[source_word + '_' + tag]
            #except:
            #    raise ValueError("Unable to encode '" + source_word + "'")
            source_vector = self.vectors[self.tokens.index(source_word)]

            for i in range(len(self.vectors)):
                #v=self.scholar.model.vectors[i]
                v=self.vectors[i]
                if spatial.distance.cosine(v,source_vector) < epsilon :
                    word_list.append(self.tokens[i])
        return "|".join(word_list)
        

if __name__ == '__main__':
    #pass
    create_fasttext_pkl(sourcefile="data/wiki.en.vec")
