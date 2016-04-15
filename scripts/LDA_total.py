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
TOKEN_MAPPINGS = [(EMAIL_REGEX, "#email"), (FILTER_REGEX, ' ')]

def tokenize_line(line):
    res = line.lower()
    for regexp, replacement in TOKEN_MAPPINGS:
        res = regexp.sub(replacement, res)

    sentence = parse(res,tokenize=True,tags=False, chunks=False, relations= False, lemmata=True).split()

    # initialize the Variables
    allowed_tags = re.compile('(NN|VB|JJ|RB)')
    stopwords = frozenset()
    min_length = 2
    max_length = 15
    result = []

    # lemmatization of the words
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

# Creating the dictionary, and bag of words corpus

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

def prep_corpus(docs, additional_stopwords=set(), no_below=5, no_above=0.5):
    print('Building dictionary...')
    dictionary = Dictionary(docs)
    # remove stopwords
    stopwords = nltk_stopwords().union(additional_stopwords)
    stopword_ids = map(dictionary.token2id.get, stopwords)
    # get ids for short words len(word)<=3
    shortword_ids = [tokenid for tokenid, word in dictionary.iteritems() if len(word.split('/')[0])<= 3]
    dictionary.filter_tokens(stopword_ids)
    dictionary.compactify()
    # get ids for short words len(word)<=3
    shortword_ids = [tokenid for tokenid, word in dictionary.iteritems() if len(word.split('/')[0])<= 3]
    dictionary.filter_tokens(shortword_ids)
    dictionary.compactify()
    # remove words that appear only once
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems()if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.compactify()
    # filter extreme values
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.compactify()

    print('Building corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus


dictionary ,corpus = prep_corpus(docs['tokens'])


MmCorpus.serialize('data/model/newsgroups.mm', corpus)
dictionary.save('data/model/newsgroups.dict')


lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=10)
lda.save('data/model/newsgroups_50.model')

