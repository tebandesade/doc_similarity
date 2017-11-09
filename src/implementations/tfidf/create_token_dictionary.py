import nltk
import string
import os
import codecs
import sys
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import json
white_spaces = []

#sys.argv[0]
#path = "md"
path = sys.argv[1]
dirs = os.listdir(path)
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' 
stop = stopwords.words('english') + list(string.punctuation)
stemmer = PorterStemmer()
token_dict = {}
lista_tokens = []
file_names= []
print str(sys.version_info[0])	+ '.'+ str(sys.version_info[1])

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = token_cleaning(tokens)
    stems = stem_tokens(tokens, stemmer)
    return stems
def token_cleaning(list_tokens):
	tokens =[]
	for item in list_tokens:
		if  (',' or '.' or '?') in item:
			if len(item)>1:
				tokens.append(item.encode('utf-8'))
		elif(item in stop or item=='!'or item=='?' or item==';' or item=='\'\'' or item =='``'):
			continue
		elif('+' in item):
			if len(item)>6:
				token_split = item.split('+')
				for index_ in token_split:
					if index_ != '' and len(index_)>1 and index_.isdigit() ==False:
						try:
							float(index_)
						except:
							tokens.append(index_.encode('utf-8'))
			else:
				continue
		##TESTING
		#elif(',' in item):
		#	print'teststeattea'
		else:
			tokens.append(item.encode('utf-8'))
	return tokens
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
	parsed_text  = re.sub(".pdf",'',parsed_text)
	parsed_text  = re.sub("(\\\)",'',parsed_text)
	parsed_text  = re.sub("'",'',parsed_text)
	
	return parsed_text.strip()

def get_tokens(file_,file_2):

	with codecs.open(file_,encoding='utf-8') as archivo:
		text = []
		for line in archivo:
			line = line.encode('utf-8')
			line = line.strip()
			if len(line)<1:
				continue

			text.append(line)
		lowers  = str(text).lower().strip()
		lowers  = lowers + file_2
		sp_clean = special_cleaning(lowers)
		tokens_  = tokenize(sp_clean.encode('utf-8'))
		token_dict[file_2] = tokens_
		file_names.append(file_2)
		lista_tokens.append(tokens_)


for file_ in dirs:
	pat_   = path +'/'+ file_

	#if os.stat(pat_).st_size <101:
	get_tokens(pat_,file_)

#print token_dict

counter = 0
for k,v in token_dict.iteritems():
	#print k	
	counter += 1 
print counter

#
count_errrr = 0 
'''
for name in file_names:
	#with open(name) as out:
		txt = lista_tokens[count_errrr]
		txt = ' '.join(txt)
		txt.strip()
		print txt
		count_errrr +=1
		'''
#tokens_list = [item for sublist in lista_tokens for item in sublist]


#print len(tokens_list)
#contador_global = Counter(tokens_list)
#print contador_global
#print len(contador_global)
#print  count
#tfidf = TfidfVectorizer( lowercase=False,stop_words='english',ngram_range=(0,3))
#tfs = tfidf.fit_transform(token_dict)
#print tfs
#np.save('token_dictionary.npy', token_dict) 



