# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import string
import sys


def token_cleaning(list_tokens):
	tokens =[]
	for item in list_tokens:
		if  (',' or '.' or '?') in item:
			if len(item)>1:
				tokens.append(item.encode('utf-8'))
		elif(item in stop or item=='!'or item=='?' or item==';' or item=='+'or item=='\'\'' or item =='``'):
			continue
		else:
			tokens.append(item.encode('utf-8'))
	return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens= token_cleaning(tokens)
    stems = stem_tokens(tokens, stemmer)
    return stems

def special_cleaning(text_):
	parsed_text  = re.sub("(\-)+",'',text_)
	parsed_text  = re.sub("(\#)+",'',parsed_text)
	parsed_text  = re.sub("(\/)+",'',parsed_text)
	parsed_text  = re.sub("(\=)+",'',parsed_text)
	parsed_text  = re.sub("(\*)+",'',parsed_text)
	parsed_text  = re.sub("(\_)+",'',parsed_text)
	parsed_text  = re.sub("&.*?;",'',parsed_text)
	parsed_text  = re.sub("(\|)+",' ',parsed_text)
	parsed_text  = re.sub("(\`)+",'',parsed_text)
	parsed_text  = re.sub("(\()+",' ',parsed_text)
	parsed_text  = re.sub("(\))+",' ',parsed_text)
	parsed_text  = re.sub("!\[",' ',parsed_text)
	parsed_text  = re.sub("(\])+",' ',parsed_text)
	parsed_text  = re.sub("(\[)+",' ',parsed_text)
	parsed_text  = re.sub("(\:)+",' ',parsed_text)
	parsed_text  = re.sub("(\.html)",'',parsed_text)
	parsed_text  = re.sub("..uploadsmain",'',parsed_text)
	parsed_text  = re.sub(".jpg",'',parsed_text)
	parsed_text  = re.sub(".png",'',parsed_text)
	parsed_text  = re.sub(".md",'',parsed_text)
	parsed_text  = re.sub("<.*>",'',parsed_text)
	parsed_text  = re.sub("(\{)+",'',parsed_text)
	parsed_text  = re.sub("(\})+",'',parsed_text)
	parsed_text  = re.sub("(\})+",'',parsed_text)
	
	return parsed_text

tokens_list =[]
def build_mapping_index_dict(dictionary_):
	counter_ = 0
	list_dictioanry_value_ = []
	list_dic_dic_   = {}
	list_name_pages  =[]
	for k, v in dictionary_.iteritems():
		list_dic_dic_[k] = counter_
		#v = " ".join(v)
		for item in v :
			tokens_list.append(item)
		list_dictioanry_value_.append(v) 
		counter_  = counter_ + 1
		list_name_pages.append(k)
	return list_dictioanry_value_ , list_dic_dic_,list_name_pages

def get_top(list_,k=None):
    score_mapping = {}
    counter = 0 
    stack = []
    for ele in list_:
        score_mapping[counter] = ele
        counter  +=1
    for key, value in sorted(score_mapping.iteritems(), key=lambda (k,v): (v,k)):
        #print "%s: %s" % (key, value)
        stack.append(key)
    for index_ in range(k):
        print list_name_pages[stack.pop()]

stop = stopwords.words('english') + list(string.punctuation)
stemmer = PorterStemmer()
read_dictionary = np.load('token_dictionary.npy').item()
list_dictioanry_value,list_dic_dic, list_name_pages = build_mapping_index_dict(read_dictionary)

tfidf_notokenize = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False,stop_words='english',ngram_range=(1,3))
tfidfscores_notokenize = tfidf_notokenize.fit_transform(read_dictionary.values())
input_ = sys.argv[1]
input_ = input_.lower()
input_ = special_cleaning(input_)
input_ = tokenize(input_)

response = tfidf_notokenize.transform([input_])
ret = tfidfscores_notokenize.dot(response.transpose())
tes = ret.todense()

get_top(tes,k=5)