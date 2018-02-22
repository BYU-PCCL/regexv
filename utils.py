import scholar.scholar as sch
from scipy import spatial
import numpy as np

### Usage from other files ###
# import utils
# v = utils.vecMaster()
# word_list = v.expand(source_word)

class vecMaster():

    def __init__(self):
        self.scholar = sch.Scholar()

    def encode(self,word):
        pass

    def decode(self,word):
        pass

    def strip_tag(self,word):
        if '_' in word:
            return word[:word.index('_')]

    def expand(self,source_words, epsilon=0.3):
        word_list = []
        for source_word in source_words:
            try:
                tag = self.scholar.get_most_common_tag(source_word)
                print "Tag is " + tag
                source_vector = self.scholar.model[source_word + '_' + tag]
            except:
                raise ValueError("Unable to encode '" + source_word + "'")

            for i in range(len(self.scholar.model.vectors)):
                v=self.scholar.model.vectors[i]
                if not np.all(v==source_vector) and spatial.distance.cosine(v,source_vector) < epsilon :
                    word_list.append(self.strip_tag(self.scholar.model.vocab[i]))
        return "|".join(word_list)
        


