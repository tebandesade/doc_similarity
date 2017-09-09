import sys
import os
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

txt_directory = 'documentation/content/md/'
directory = os.listdir(txt_directory)
counter = 0
vocab = []
sentences = 0
oraciones= []
paths = []
jsons = []
nltkc_docs = []
import string


dictionary_docs = {}

stop = stopwords.words('english') + list(string.punctuation)


docs_string_nltk = ""
for file_ in directory:
	path = os.path.join(str(txt_directory),str(file_))
	#Ignores file that are less than 100k. Looks like the files that are less, don't have info
	if os.stat(path).st_size >=100:
		#print (path) if you want to se which pages are not included
	#else:
		with open(path) as arch: ## open('out_clean/'+file_,'w') as out:
			data_json = {'id':file_,'text':None}
			counter = counter +1
			paths.append(path)
			texto_pagina = ''
			#nltk_doc = []
			doc_texto = []
			for line in arch:
				sentences +=1
				line = line.strip()
				if len(line)!=0:
					#print(line)
					#print('##################################')
					parsed_text  = re.sub("(\-)+",'',line)
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
					#parsed_text  = re.sub("\]\((\.\.uploads.?\)",'',parsed_text)
					parsed_text  = re.sub("[0-9]\.",'',parsed_text)
					parsed_text  = re.sub("(\{)+",'',parsed_text)
					parsed_text  = re.sub("(\})+",'',parsed_text)
					parsed_text  = parsed_text.strip()
					parsed_text  = " ".join(parsed_text.split())
					sent_txt = nltk.sent_tokenize(parsed_text)
					sent_temp = []
					for sent in sent_txt:
						sent = sent.lower()
						oraciones.append(sent)
						txt = nltk.word_tokenize(sent)
						#texto.append(txt)
						#print(txt)
						for word in txt:
							if word not in stop:
							#print(word)
								vocab.append(word)
								sent_temp.append(word)
					#sent_temp_string = ",".join(sent_temp)
					texts_weird = [ word for word in sent_temp if word not in stop]
					#texts_weird_string = " ".join(texts_weird)

					#texts_weird =  ''.join(texts_weird)
					#print(texts_weird)
					if(texts_weird):

						#out.write(texts_weird_string)
						#out.write('\n')
						doc_texto.append(texts_weird)

					#nltk_doc.append(texts_weird)

			#print("#########################FLATTTTT")
			#print(flat_list)
			flat_list = [item for sublist in doc_texto for item in sublist]
			nltkc_docs.append(flat_list)
			#dictionary_docs[file_] = doc_texto
			#data_json['text'] = doc_texto
			#jsons.append(data_json)
			#for line in arch:
				#print(line)

print('Pages: ',counter)
print('Vocabulary: ',len(vocab))
print('Sentences: ',len(oraciones))
print('Test sentence: ', oraciones[1])
print('Length of sentence: ', len(oraciones[1]))

#text = " ".join(texto)
flat_list = [item for sublist in nltkc_docs for item in sublist]
print(len(flat_list))
text = nltk.Text(vocab)
#Falla si se paasa nltkdocs
#text_pos = nltk.pos_tag(text)
#text_pos = nltk.pos_tag(text)
#print(text_pos)
##print(text.concordance('this'))
fdist_2 = nltk.FreqDist(text)
print(fdist_2)
fdist = nltk.FreqDist(vocab)
print(fdist)
#print(fdist.most_common(50))

wordnet_lemmatizer = WordNetLemmatizer()
lemmas = []
#for txt in text_pos:
#	print(txt[0])
#	if txt[1] in [ 'a', 's', 'r', 'n', 'v']:
#		print(txt[0])
#		lemmas.append(wordnet_lemmatizer.lemmatize(txt[0],pos=txt[1]))
#		print(wordnet_lemmatizer.lemmatize(txt[0],pos=txt[1]))
#	else:
#		print(txt[0])
#		lemmas.append(wordnet_lemmatizer.lemmatize(txt[0]))
#		print(wordnet_lemmatizer.lemmatize(txt[0]))
#
#fdist_lemma = nltk.FreqDist(lemmas)
#print(fdist_lemma)

from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
no_features = 1000

documents = nltkc_docs

print (len(documents))
#print(len(text_pos))

print (type(documents))
print(nltkc_docs[0][0])
#nltk_Docs_str = ''.join(documents)
print("##############")
print(type(nltkc_docs[0]))
#nltk_Docs_str = ' '.join(documents)
#print(str(nltk_Docs_str))
#dictionary = corpora.Dictionary(documents)
#dictionary.save('/tmp/unitydocs.dict')


#print(dictionary_docs['UNetHostMigration.md'])
#dictionary = corpora.Dictionary(documents)
#np.save('dictionary_unity.clean.npy',dictionary_docs)


# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print(len(tfidf_feature_names))
'''
no_topics = 20
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

def display_topics(model, feature_names, no_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print(topic_idx)
		print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
#display_topics(lda, tf_feature_names, no_top_words)
'''


#for doc in nltkc_docs:
	#print(doc)


#for pa in paths:
#	print(pa)
#import json
#with open('unity_clean.json','w') as f:
#	for item in jsons:
#		json.dump(item,f)
#		f.write(os.linesep)
#np.save('dictionary_json.unity.clean.npy',jsons)
#print (data_json['GraphicsOverview.md'])
###TO DOOOOOOOO PUT DOCS IN DATABASE
#Todo
#Clean whitespaces in files check
#Leave a sentence per line check
# remove
	# //   x
	# { }  maybe not
	# ```` x
	# ---- x
	#1.    x
	# === x
	#[](../uploads/...) x
	# __ASDASDAS__ (but solo quitar __) x
	# -
	#|__ASdasdasdas ASDa__|x
	#*dsadasdas* x
	#[Build Setting](xboxone-buildsettings) analyze if it gives important info , it does so will have to remove brakets
	#<span class="inspector">Application Manifest</span>. Check if the class changes in other, to see if it is label
