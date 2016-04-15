from glob import glob
import re
import string
import funcy as fp
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import pandas as pd
from pattern.en import parse
import logging
logging.basicConfig(filename="logging.txt", format='%(asctime)s : %(levelname)s : %(message)s',filemode ="w", level=logging.INFO)


# quick and dirty...
EMAIL_REGEX = re.compile(r"[a-z0-9\.\+_-]+@[a-z0-9\._-]+\.[a-z]*")
FILTER_REGEX = re.compile(r"[^a-z '#]")
TOKEN_MAPPINGS = [(EMAIL_REGEX, "email"), (FILTER_REGEX, ' ')]

def tokenize_line(line):
    res = line.lower()
    for regexp, replacement in TOKEN_MAPPINGS:
        res = regexp.sub(replacement, res)
	
    sentence = parse(res,tokenize=True,tags=False, chunks=False, relations= False, lemmata=True).split()
    
	#initialize the Variables
    allowed_tags = re.compile('(NN|VB|JJ|RB)')
    stopwords = frozenset()
    min_length = 2
    max_length = 15
    result = []
    
    try:
        sentence = sentence[0]
    except IndexError:
        pass 
    
    for token, tag, lemma in sentence:
        if min_length <= len(lemma) <= max_length and lemma not in stopwords:
            if allowed_tags.match(tag):
                lemma += "/" + tag[:2]
                result.append(lemma.encode('utf8'))
    res = result
    logging.info("That's how res looks %s" %res)
    return res
	#return res.split()
    
def tokenize(lines, token_size_filter=2):
    tokens = fp.mapcat(tokenize_line, lines)
    return [t for t in tokens if len(t) > token_size_filter]
    

def load_doc(filename):
    # Slash for linux and double backslash for windows
    group, doc_id = filename.split('\\')[-2:]
    with open(filename) as f:
        doc = f.readlines()
    logging.info("logging in %s in doc %s" %(group, doc_id))
    return {'group': group,
            'doc': doc,
            'tokens': tokenize(doc),
            'id': doc_id}


docs = pd.DataFrame(list(map(load_doc, glob('data/20news-bydate-train/*/*')))).set_index(['group','id'])
docs.to_csv("data/model/docs.csv")


"""# # Creating the dictionary, and bag of words corpus

# In[ ]:

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

def prep_corpus(docs, additional_stopwords=set(), no_below=5, no_above=0.5):
    print('Building dictionary...')
    dictionary = Dictionary(docs)
    #remove stopwords and words that appear only once 
    stopwords = nltk_stopwords().union(additional_stopwords)
    stopword_ids = map(dictionary.token2id.get, stopwords)
    #once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems()if docfreq == 1]
    dictionary.filter_tokens(stopword_ids)
    dictionary.compactify()
    # filter extreme values 
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.compactify()

    print('Building corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus


# In[ ]:

dictionary, corpus = prep_corpus(docs['tokens'])


# In[ ]:

print corpus


# # Save the dictionary and the corpus

# In[ ]:

MmCorpus.serialize('model/newsgroups.mm', corpus)
dictionary.save('model/newgroups.dict')


# # Fitting the LDA model

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=10)\nlda.save('model/newsgroups_50.model')")

"""