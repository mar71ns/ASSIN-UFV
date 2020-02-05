#from nltk.examples.pt import *
import nltk
#nltk.download('rslp')
from nltk.corpus import machado, mac_morpho, floresta, genesis
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.util import bigrams
from nltk.misc import babelize_shell
from nltk import ngrams
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import Tree
import re
import pydash
from pydash import split, deburr, separator_case, flatten_deep, pad, push, get, set_
from pydash import intersection, union, min_
from spacy.tokenizer import Tokenizer
import spacy
import pt_core_news_sm
import numpy as np
from scipy import spatial
from spacy.pipeline import DependencyParser
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
nlp = spacy.load('pt_core_news_sm')
stemmer = nltk.stem.RSLPStemmer()
from gensim.models import KeyedVectors
#word_vectors = KeyedVectors.load_word2vec_format('./Vectors/glove_s300.txt')
#word_vectors = KeyedVectors.load_word2vec_format('skip_s1000.txt')
word_vectors = KeyedVectors.load_word2vec_format('glove_s50.txt.txt')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------RELATÓRIO---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#######################################################################
#####################------PRINT ÁRVORE DA FRASE------################
#######################################################################
def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
    # def to_nltk_tree(node):
    #     if node.n_lefts + node.n_rights > 0:
    #         return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    #     else:
    #         return tok_format(node)

def print_tree_from_sent(f1,f2):

    doc = nlp(f1)

    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


#############################################################
####################------PRINT FEATURE------################
#############################################################
def print_intersection_detail(title,f1,f2):
    #Retorna números
    print('\n')
    print(pad(title,50,'-'))
    print('\n',f1,'\n\n',f2,'\n')
    print(intersection(f1,f2))
    print('\n')

############################################################
##############------INTERSECTION LENGHT------################
#############################################################
def len_intersection(f1,f2):
    return len(intersection(f1,f2))

############################################################
##############------JACCARD SIMILARITY------################
#############################################################
def jaccard_similarity(f1,f2):
    # print(union(f1,f2),"\n")
    # print(len(union(f1,f2)),"\n\n")
    div_by = len(union(f1,f2))
    if div_by == 0:
        return 0
    return len(intersection(f1,f2))/div_by

############################################################
##############------OVERLAP SIMILARITY------################
#############################################################
def overlap_similarity(f1,f2):
    # print(len(intersection(f1,f2)))
    # print(min_([len(f1),len(f2)]))
    # print(len(intersection(f1,f2))/min_([len(f1),len(f2)]))
    div_by = min_([len(f1),len(f2)])
    if div_by == 0:
        return 0
    return len(intersection(f1,f2))/div_by

############################################################
##############------COSINE SIMILARITY------################
#############################################################
def cos_similarity(f1,f2):
    if(len(f1)==0 or len(f2)==0):
        return 0
    f1 = (' '.join(map(str, f1)))
    f2 = (' '.join(map(str, f2)))
    count_vectorizer = CountVectorizer()
    try:
        sparse_matrix = count_vectorizer.fit_transform([f1,f2])
    except Exception as e:
        #print(e)
        return 0
    #print(sparse_matrix)
    doc_term_matrix = sparse_matrix.todense()
    # df = pd.DataFrame(doc_term_matrix,
    #                   columns=count_vectorizer.get_feature_names(),
    #                   index=['f1', 'f2'])
    # cos=cosine_similarity(df)
    # print(cos)

    # # Compute Cosine Similarity
    cos=cosine_similarity(doc_term_matrix[0], doc_term_matrix[1])

    return (cos)

############################################################
##############------SOFT COSINE SIMILARITY------################
#############################################################
def soft_cos_similarity(f1,f2):
    count_vectorizer = CountVectorizer()
    f1 = (' '.join(map(str, f1)))
    f2 = (' '.join(map(str, f2)))
    #print(' '.join(map(str, f1)))
    sparse_matrix = count_vectorizer.fit_transform([f1,f2])
    #print(sparse_matrix)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,
                      columns=count_vectorizer.get_feature_names(),
                      index=['f1', 'f2'])
    # Compute Cosine Similarity
    cos=cosine_similarity(doc_term_matrix[0], doc_term_matrix[1])
    #print(cos)
    return (cos)

############################################################
#################------DICE SIMILARITY------################
############################################################
def dice_similarity(f1,f2):
    if(len(f1)==0 or len(f2)==0):
        return 1
    try:
        count_vectorizer = CountVectorizer()
        f1 = (' '.join(map(str, f1)))
        f2 = (' '.join(map(str, f2)))
        sparse_matrix = count_vectorizer.fit_transform([f1,f2])
        doc_term_matrix = sparse_matrix.todense()
        # Compute DICE Similarity
        dice=distance.dice(doc_term_matrix[0], doc_term_matrix[1])
        #print(dice)
        return (dice)
    except Exception as e:
        return 1

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#######################################################################
##############------RETORNA TOKENS DA FRASE------################
#######################################################################
def get_tokens_from_sent(f1):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    return f1

#######################################################################
##############------RETORNA TOKENS DAS FRASES------################
#######################################################################
def get_tokens_from_2_sents(f1,f2):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    f2 = word_tokenize(deburr(separator_case(f2," ")))
    return f1,f2

#######################################################################
##############------RETORNA PALAVRAS DA FRASE------################
#######################################################################
def get_words_from_sent(f1):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    #Remove números
    f1 = [x for x in f1 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    return f1

#######################################################################
##############------RETORNA PALAVRAS DAS FRASES------################
#######################################################################
def get_words_from_2_sentS(f1,f2):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    f2 = word_tokenize(deburr(separator_case(f2," ")))
    #Remove números
    f1 = [x for x in f1 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    f2 = [x for x in f2 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    return f1,f2

#######################################################################
##############------NUM INTERSEÇÃO ENTRE NUMERAIS------################
#######################################################################
def get_numbers_from_sent(f1):
    #Retorna números
    f1 = re.findall(r"[-+]?\d*\.\d+|\d+",f1)
    return f1

#######################################################################
##############------NUM INTERSEÇÃO ENTRE NUMERAIS------################
#######################################################################
def get_numbers_from_2_sents(f1,f2):
    #Retorna números
    f1 = re.findall(r"[-+]?\d*\.\d+|\d+",f1)
    f2 = re.findall(r"[-+]?\d*\.\d+|\d+",f2)
    return f1,f2

#######################################################################
##############------RETORNA DA FRASE------################
#######################################################################
def get_obj_pos_from_sent(f1):

    doc = nlp(f1)
    pos = {"ADJ":[],"ADP":[],"ADV":[],"AUX":[],"CONJ":[],"CCONJ":[],"DET":[],"INTJ":[],"NOUN":[],"NUM":[],"PART":[],"PRON":[],"PROPN":[],"PUNCT":[],"SCONJ":[],"SYM":[],"VERB":[],"X":[]}
    for token in doc:
        list=[]
        id = (str(token.pos_))
        # if get(pos, id) is not None:
        #     list = get(pos, id)
        list = push(list, (str(token.orth_)).lower())
        set_(pos, id, list)
    return pos

#######################################################################
##############------RETORNA LEMMAS DA FRASE------################
#######################################################################
def get_lemmas_from_sent(f1):
    s1 = "",""
    #Retorna Lemmas
    for token in nlp(f1):
        s1 += "" if token.pos_ == "PUNCT" else token.lemma_+ " "
    #Tokens, Lowercase sem ascentos
    f1 = word_tokenize(deburr(s1.lower()))
    return f1

#######################################################################
##############------RETORNA LEMMAS DAS FRASES------################
#######################################################################
def get_lemmas_from_2_sents(f1,f2):
    s1,s2 = "",""

    #Retorna Lemmas
    for token in nlp(f1):
        s1 += "" if token.pos_ == "PUNCT" else token.lemma_+ " "
    for token in nlp(f2):
        s2 += "" if token.pos_ == "PUNCT" else token.lemma_+ " "
    #Tokens, Lowercase sem ascentos
    f1 = word_tokenize(deburr(s1.lower()))
    f2 = word_tokenize(deburr(s2.lower()))

    return f1,f2

#######################################################################
##############------RETORNA TAGGER COMPLETA DA FRASE------################
#######################################################################
def get_tagger_from_sent(f1):
    f1 = deburr(f1.lower())

    s1= []
    #Retorna Lemmas
    for token in nlp(f1):
        s1 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s1.append(token.text + "|" +token.tag_)
    return s1

#######################################################################
##############------RETORNA TAGGER COMPLETA DAS FRASES------################
#######################################################################
def get_word_tag_from_2_sents(f1,f2):
    s1,s2 = [],[]
    #Retorna Lemmas
    nlp = spacy.load('pt_core_news_sm')
    for token in nlp(f1):
        s1 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s1.append(deburr(token.text.lower()) + "|" +token.tag_)
    for token in nlp(f2):
        s2 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s2.append(deburr(token.text.lower()) + "|" +token.tag_)
    nlp = spacy.load('en_core_web_sm')
    for token in nlp(f1):
        s1 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s1.append(deburr(token.text.lower()) + "|" +token.tag_)
    for token in nlp(f2):
        s2 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s2.append(deburr(token.text.lower()) + "|" +token.tag_)

    return s1,s2

#######################################################################
##############------RETORNA WORD TAG DA FRASE------################
#######################################################################
def get_word_tag_from_sent(f1):
    s1= []
    for token in nlp(f1):
        s1 if (token.pos_ == "PUNCT" or token.pos_ == "SPACE") else s1.append(token.text + "|" +token.tag_)
    return s1

#######################################################################
##############------RETORNA WORD POS DAS FRASES------################
#######################################################################
def get_word_pos_from_2_sents(f1,f2):
    doc = nlp(f1)
    dependencies=[]
    for token in doc:
        if (token.pos_ == "PUNCT" or token.pos_ == "SPACE"):
            continue
        word_pos = deburr(token.text.lower()) +" "+ token.pos_
        push(dependencies,word_pos)
    s1 = dependencies
    dependencies=[]

    doc = nlp(f2)
    for token in doc:
        if (token.pos_ == "PUNCT" or token.pos_ == "SPACE"):
            continue
        #print(token.text, token.pos_)
        word_pos = deburr(token.text.lower()) +" "+ token.pos_
        push(dependencies,word_pos)
    s2 = dependencies

    return s1,s2

#######################################################################
##############------RETORNA STEMS DA FRASE------################
#######################################################################
def get_stems_from_sent(f1,f2):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    #Remove números
    f1 = [stemmer.stem(x) for x in f1 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    return f1

#######################################################################
##############------RETORNA STEMS DAS FRASES------################
#######################################################################
def get_stems_from_2_sents(f1,f2):
    #Tokens, Lowercase sem ascentos, remove pontuação
    f1 = word_tokenize(deburr(separator_case(f1," ")))
    f2 = word_tokenize(deburr(separator_case(f2," ")))
    #Remove números
    f1 = [stemmer.stem(x) for x in f1 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    f2 = [stemmer.stem(x) for x in f2 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    return f1,f2

#######################################################################
##############------RETORNA DEPENDENCY PARSER DAS FRASES------################
#######################################################################
def get_syntactic_dependency_from_sent(f1):

    doc = nlp(f1)
    dependencies=[]
    for token in doc:
        push(dependencies,(token.head.text, token.text, token.dep_))

    return dependencies

#######################################################################
##############------RETORNA ENTIDADES DA FRASE------################
#######################################################################
def get_entities_from_sent(f1):
    entities={}

    nlp = spacy.load('pt_core_news_sm')
    doc = nlp(f1)
    for ent in doc.ents:
        list=[]
        if get(entities,str(ent.label_)) is not None:
            list = get(entities,str(ent.label_))
        list = push(list,str(ent))
        set_(entities,str(ent.label_),list)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(f1)
    for ent in doc.ents:
        if str(ent.label_) == 'TIME' or str(ent.label_) == 'ORDINAL' or str(ent.label_) == 'DATE' or  str(ent.label_) == 'CARDINAL' or  str(ent.label_) == 'MONEY' or  str(ent.label_) == 'QUANTITY':
            list=[]
            if get(entities,str(ent.label_)) is not None:
                list = get(entities,str(ent.label_))
            list = push(list,str(ent))
            set_(entities,str(ent.label_),list)

    return entities

#######################################################################
##############------RETORNA PROPORÇÃO TAMANHOS DA FRASE------################
#######################################################################
def get_len_proportion_from_2_sents(f1,f2):
    if len(f1)==0 or len(f2)==0:
        return 0
    if len(f1) > len(f2):
        return len(f1)/len(f2)
    else:
        return len(f2)/len(f1)

#######################################################################
##############------RETORNA PALAVRAS - ENTIDADES DA FRASE------################
#######################################################################
def get_word_entitie_from_sent(f1):
    doc = nlp(f1)
    #print(doc)
    entities=[]
    for ent in doc.ents:
        tuple = str(ent)+ " " +str(ent.label_)
        push(entities,tuple)
    return entities

#######################################################################
##############------RETORNA PALAVRAS - ENTIDADES DA FRASE------################
#######################################################################
def get_type_entities_from_sent(f1):
        doc = nlp(f1)
        dependencies=[]
        for token in doc:
            print(token.text, token.pos_)
            # if token.dep_=="ROOT":
            #     continue
            # push(dependencies,(token.head.text, token.text, token.dep_))

        for chunk in doc.noun_chunks:
            print(chunk)
            #print(chunk.text, chunk.root.dep_)
            print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)

        # print(pad("",105,'-'))
        # for word in doc:
        #     if word.dep_ == 'nsubj':
        #         print('Subject:', word.text)
        #     if word.dep_ == 'dobj':
        #         print('Object:', word.text)

        return dependencies

#################################################################################################
##############------RETORNA LEN INTERSECTION DE LISTA COM STRINGS E A FRASE------################
#################################################################################################
def process_adverb(adverbs,f1,f2):
    f1 = f1.lower()
    f2 = f2.lower()
    doc = nlp(f1)
    dependencies=[]
    len1 = 0
    len2 = 0
    adv1 = []
    adv2 = []
    for item in adverbs:
        item = item.lower()
        if item in f1:
            len1 = len1 + 1
            push(adv1,str(item))
        if item in f2:
            len2 = len2 + 1
            push(adv2,str(item))

    return [len1, len2, abs(len1 - len2), len_intersection(adv1, adv2)]

#################################################################################################
##############------PRINT ADVERBS RESULT------################
#################################################################################################
def print_adverbs_result(ft,res):
    print('{:40s} {:3d}'.format(ft + "Len(f1)", res[0]))
    print('{:40s} {:3d}'.format(ft + "Len(f2)", res[1]))
    print('{:40s} {:3d}'.format(ft + "Abs Diff", res[2]))
    print('{:40s} {:3d}'.format(ft + "Len Inter", res[3]))

#################################################################################################
##############------GENSIM WV MOST SIMILAR------################
#################################################################################################
#return a list of words that contain gensim vectors and they respectives similars
def get_wv_most_similar(sent):
    wd = []
    similar = []

    if not isinstance(sent, list):
        sent = get_tokens_from_sent(sent)

    for tk in sent:
        try:
            result = word_vectors.most_similar(positive=[tk])

            for item in result:
                similar.append(item[0])
            wd.append(tk.lower())
        except Exception as e:
            pass

    return wd,similar

#################################################################################################
##############------GENSIM WM DISTANCE------################
#################################################################################################
#return a list of words that contain gensim vectors and they respectives similars
def get_gensim_sent_distance(sent1, sent2):
    try:
        distance = word_vectors.wmdistance(sent1,sent2)
        #print("{:.4f}".format(distance))
        return distance
    except Exception as e:
        print(e)
        return 0

#################################################################################################
##############------GENSIM COS SIMILARITY------################
#################################################################################################
#return a list of words that contain gensim vectors and they respectives similars
def get_gensim_cos_similarity(sent1, sent2):
    try:
        similarity = word_vectors.n_similarity(sent1,sent2)
        #print("{:.4f}".format(similarity))
        return similarity
    except Exception as e:
        print(e)
        return 0

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
