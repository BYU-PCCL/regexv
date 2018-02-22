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
        self.tokens = np.atleast_1d(data['tokens'])
        self.vectors = data['vectors']

    def strip_tag(self,word):
        if '_' in word:
            return word[:word.index('_')]

    def neighbor_expansion(self, source_words, epsilon=0.35, distance_metric = 'cosine'):
        return source_words
        # source_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in source_words])
        # distances = spatial.distance.cdist(self.vectors,np.atleast_2d(source_vectors), distance_metric)[:,0]
        # return np.squeeze(self.tokens[np.argwhere(distances<epsilon)])


    def mahalanobis_expansion(self, source_words, epsilon=None, sigma=0.001):
        source_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in source_words])
        c = np.cov(source_vectors.T)
        c += sigma * np.identity(c.shape[0])
        c = np.linalg.inv(c) 

        def mahalanobis_squared(u, v, VI=c):
            delta = u - v
            return np.dot(np.dot(delta, VI), delta)
        centroid = np.atleast_2d(np.mean(source_vectors, axis=0))
        distances = spatial.distance.cdist(self.vectors, centroid, metric=mahalanobis_squared)

        print(source_vectors[:,:10])
        print(centroid.shape)
        print(centroid)
        print(distances.shape)
        print(distances)
        if epsilon is None:
             d = spatial.distance.cdist(source_vectors, centroid, metric=mahalanobis_squared)
             print("HERE")
             print(d.shape)
             print(d)
             epsilon = np.max(d)
        print(epsilon)
        return np.squeeze(self.tokens[np.argwhere(distances<epsilon)])


if __name__ == '__main__':
    v=vecMaster()
    #print(v.mahalanobis_expansion(['beautiful', 'gorgeous']))
    print(v.neighbor_expansion(['beautiful', 'gorgeous']))
