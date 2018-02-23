# import scholar.scholar as sch
from scipy import spatial
import numpy as np
import pickle as pkl


### Usage from other files ###
# import utils
# v = utils.vecMaster()
# word_list = v.expand(source_words, expansion_method, epsilon)

# def create_vector_object(sourcefile="data/fasttext.wiki.en.vec", destfile="data/fasttext.en", truncate=None):

### NOTE: run this function using python3
# otherwise the dict file is unusable
def create_fasttext_pkl(sourcefile="fasttexttrunc", destfile="data/fasttext.en", truncate=None):
    f = open(sourcefile, "r")
    firstline = f.readline()
    # vector_dict={}
    # token_dict={}
    tokens = []
    vectors = []
    for line in f:
        token = line[:line.index(' ')]
        vector_string = line[line.index(' ') + 1:]
        vector = np.fromstring(vector_string, sep=' ')
        # token_dict[vector_string] = token
        # vector_dict[token] = vector
        tokens.append(token)
        # tokens.append(unitcode(token.decode('utf-8',errors='ignore')))
        vectors.append(vector)
    vectors = np.vstack(vectors)
    data = {}
    data['tokens'] = tokens
    data['vectors'] = vectors
    f = open(destfile + '.pkl', 'wb')
    pkl.dump(data, f, protocol=4)
    f.close()


class vecMaster():

    def __init__(self, sourcefile='data/fasttext.en.pkl'):
        with open(sourcefile, 'rb') as myfile:
            data = pkl.load(myfile)
        self.token_list = data['tokens']
        self.tokens = np.atleast_1d(self.token_list[:50000])
        self.vectors = data['vectors'][:50000]

    def validate(self, word_list):
        valid_words = word_list.copy()
        for w in word_list:
            if w not in self.tokens:
                if ' ' in w:
                    for sub_w in w.split(' '):
                        if sub_w not in self.tokens:
                            print("Word " + w + " not found in vector model. Omitting...")
                            valid_words.remove(w)
        return valid_words

    def neighbor_expansion(self, source_words, epsilon=0.35, distance_metric='cosine', k=None):

        source_words = self.validate(source_words)
        #source_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in source_words])
        sv = []
        for w in source_words:
            #if multiword, then average them
            if ' ' in w:
                words = w.split(' ')
                phrase_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in words])
                sv.append(np.mean(phrase_vectors,axis=0)) 
            else:
                #otherwise, take the vector
                sv.append(self.vectors[np.squeeze(np.argwhere(self.tokens==w))])
        source_vectors = np.vstack(sv)

        distances = spatial.distance.cdist(self.vectors, source_vectors, distance_metric)[:,0]

        if k is not None:
            # find the k nearest
            inds = np.argsort( distances )
            return np.array( self.tokens[ inds[0:k] ] )
        else:
            return np.squeeze(self.tokens[np.argwhere(distances < epsilon)])

    def mahalanobis_expansion(self, source_words, epsilon=0.25, k=None, sigma=0.00001):
        source_words = self.validate(source_words)
        source_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in source_words])

        c = np.cov(source_vectors.T)
        c += sigma * np.identity(c.shape[0])
        c = np.linalg.inv(c)
        #c = np.linalg.pinv(c)

        def mahalanobis_squared(u, v, VI=c):
            delta = u - v
            return np.dot(np.dot(delta, VI), delta)

        centroid = np.atleast_2d(np.mean(source_vectors, axis=0))
        distances = spatial.distance.cdist(self.vectors, centroid, metric=mahalanobis_squared)[:, 0]

        if k is not None:
            # find the k nearest
            inds = np.argsort( distances )
            return np.array( self.tokens[ inds[0:k] ] )
        else:
            # find anything within radius epsilon (scaled by mean distance)
            epsilon = epsilon * np.mean(distances)
            return np.squeeze(self.tokens[np.argwhere(distances <= epsilon)])

    def naive_centroid_expansion(self, source_words, epsilon=0.25, distance_metric='cosine', k=None):
        source_words = self.validate(source_words)
        source_vectors = np.array([self.vectors[np.squeeze(np.argwhere(self.tokens == w))] for w in source_words])

        centroid = np.atleast_2d(np.mean(source_vectors, axis=0))
        distances = spatial.distance.cdist(self.vectors, centroid, distance_metric)[:, 0]

        if k is not None:
            # find the k nearest
            inds = np.argsort( distances )
            return np.array( self.tokens[ inds[0:k] ] )
        else:
            # find anything within radius epsilon (scaled by mean distance)
            epsilon = epsilon * np.mean(distances)
            return np.squeeze(self.tokens[np.argwhere(distances <= epsilon)])


if __name__ == '__main__':
    v=vecMaster()
    #print(v.neighbor_expansion(['beautiful', 'gorgeous', 'handsome'], k=30))
    #print(v.mahalanobis_expansion(['beautiful', 'gorgeous', 'handsome'], k=30))
    #print(v.neighbor_expansion(['beautiful', 'gorgeous', 'handsome'], k=30))
    #print(v.mahalanobis_expansion(['beautiful', 'gorgeous', 'handsome', 'studly','hot'],k=20))
    #print(v.neighbor_expansion(['france', 'germany','guatemala'], k=30))
    #print(v.mahalanobis_expansion(['france', 'germany','guatemala'], k=30))
    #print(v.neighbor_expansion(['red', 'green','blue','yellow','ruby','orange','maroon'],k=20))
    #print(v.mahalanobis_expansion(['red', 'green','blue','yellow','ruby','orange','maroon'],k=20))
    # print(v.neighbor_expansion(['beautiful', 'gorgeous']))
    # print(v.neighbor_expansion(['red', 'green','blue']))
    #print(v.neighbor_expansion(['idiot', 'jerk','stupid','dumb','fat','imbecile','imbecilic','sadistic'], k=30))
    #print(v.neighbor_expansion(['idiot', 'jerk','stupid','dumb','fat','imbecile','imbecilic','sadistic'], k=30))
    #print(v.mahalanobis_expansion(['idiot', 'jerk','stupid','dumb','fat','imbecile','imbecilic','sadistic'], k=30))
    print(v.neighbor_expansion(['clever guy'], k=30))
    #print(v.mahalanobis_expansion(['genius', 'prodigy','innovator'], k=30))

