#!/usr/bin/env python

# Trying out Gensim Tutorial #1 http://radimrehurek.com/gensim/tut1.html


import logging
import re
import bz2

import gensim
from gensim import corpora,models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# id2word = gensim.corpora.Dictionary.load_from_text('nasa-mars-teaching.txt')
documents=open('nasa-mars-teaching.txt').readlines()
documents = [doc.strip() for doc in documents if doc.strip()]

stoplist = set('for a of the and to in is it an as at than that will be we on but with are or'.split())
# TODO: remove punctuation
texts = [[word.replace('.','') for word in document.lower().split() if word not in stoplist]
         for document in documents if document.strip()]
# text is a list of lines of a list of words
all_tokens = sum(texts, [])     # a flat list of words

tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]
dictionary = corpora.Dictionary(texts)
dictionary.save('nasa-mars.dict')
logging.info("dictionary.token2id=%s" % dictionary.token2id)

# Get tweets with:
# curl 'http://search.twitter.com/search.atom?q=nasa+mars&lang=en&rpp=100&result_type=recent' > nasa-mars-twitter.xml
tweetxml=open('nasa-mars-twitter.xml').read()
tweets = re.findall('<title>(.*?)</title>', tweetxml)[1:]

vec0 = dictionary.doc2bow(tweets[0].lower().split())
logging.info("vec0=%s" % vec0)  #
logging.info("words vec0=%s" % ( ["%s=%d" % (dictionary[n], count) for (n,count) in vec0]))


vec3 = dictionary.doc2bow(tweets[3].lower().split())
logging.info("vec3=%s" % vec3)
logging.info("words vec3=%s" % ( ["%s=%d" % (dictionary[n], count) for (n,count) in vec3]))
# INFO : words vec3=['mars=1', 'planet=1', 'red=1', 'from=1']

# bag the original text, not the tweets yet.

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('texts.mm', corpus)
logging.info("list corpus=%s" % list(corpus))


# http://radimrehurek.com/gensim/tut2.html

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    logging.info("tfidf doc=%s" % ["%s=%d" % (dictionary[n], 100 * val) for (n, val) in doc])

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # TODO: 200-500 are typical
corpus_lsi = lsi[corpus_tfidf]  # logs topic#0(1.650): ...; topic#1(1.334): ...
# logging.info("lsi.print_topics(2):")
# lsi.print_topics(2)             # this 'print' actually logs at INFO level; what's '2' for?

# We an keep feeding new docs to lsi model with .add_documents(some
# other tfidf corpus) and be told to "forget" older entries

lsi.save('model.lsi')

for n, doc in enumerate(corpus_lsi):
    logging.info("lsi %d %s %.40s" % (n, doc, documents[n]))

# other models

lda_model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=5) # num should be larger
lda_model.print_topics()

hdp_model = models.hdpmodel.HdpModel(corpus, id2word=dictionary) # no num_topics kwarg, I get 20
lda_model.print_topics()                                         # we could lower this to 5, for display

import pdb; pdb.set_trace()


### http://radimrehurek.com/gensim/tut3.html
