
from gensim.models import Doc2Vec
import numpy as np
from gensim.parsing import preprocessing as pre
import os
import collections
import smart_open
import random
import gensim
import nltk
from nltk.corpus import stopwords
import string
from nltk.util import ngrams


path_ = "../out_clean"

##Double preprocessing
def read_corpus_(fname,l):
	tokens_only=False
	lines= []
	with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
		for i, line in enumerate(f):
		# For training data, add tags
			#print(line)
			line = line.strip()
			line = pre.strip_tags(line)
			line = pre.strip_punctuation(line)
			line = pre.strip_numeric(line)
			line = line.strip()
			#line_token = nltk.word_tokenize(line)
			#bigrams  = ngrams(line_token,2)

			lines.append(line.split())
			#doc.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line),[i]))
	text = [j for t in lines for j in t]
	#print(text)
	return gensim.models.doc2vec.TaggedDocument(text,[l])
	#documents.append(doc)



files     = os.listdir(path_)
i = 0
documents = []	
dictionary = {} 

for file_ in files:
	file_path = os.path.join(path_,file_)
	doc_2v = read_corpus_(file_path,i)
	dictionary[i] = file_
	i=i+1
	documents.append(doc_2v)

model   = Doc2Vec(documents,size=150,hs=1,min_count=2,workers=4, iter=20,dbow_words=1,dm=0)

stop = stopwords.words('english') + list(string.punctuation)
input_ = 'We are trying to use Host Migration with online matchmaker.To make it simple now we are using it now with "Show GUI".Basically we added a custom NetworkMigrationManager, where we only overrided OnClientDisconnectedFromHost, where we call the base function and set a flag to disable any message sending after migration (for testing). See the attached file: HostMigration.cs. We start with 3 players, then the server quits, and host migration happens between the 2 remaining machines by using the UI buttons. It seems like that it happens successfully, there will be a new server, and the another client receives this log: NetworkClient Reconnect::ffff:52.28.11.218:5054 UnityEngine.Networking.NetworkMigrationManager:OnGUI(). But when we try to send the first message (through a chat), we get this error: Send command attempted with no client running [client=hostId: 0 connectionId: 1 isReady: False channel count: 2].UnityEngine.Networking.NetworkBehaviour:SendCommandInternal(NetworkWriter, Int32, String) NetworkPlayer:CallCmdServerChatMessage(PlayerId, String) This is the point where we are stuck...We received this Send command attempted with no client running... message all the time when we try to send any message. What could be the problem? '
test = input_.lower()
test = pre.strip_punctuation(test)
test = pre.strip_tags(test)
test = pre.strip_numeric(test)
test_final = [i for i in nltk.word_tokenize(test) if i not in stop]
bigrams  = ngrams(test_final,2)
list_input = []
#for bi in bigrams:
#	for word in bi:
#		list_input.append(word)
#print(list_input)
#for i in range(10):
#print(help(model))

#test_vector = model[test_final]
vector_test = model.infer_vector(test_final,steps=10000)
#print(model.docvecs[test_final])
#similar_by_vector = model.similar_by_vector(vector_test)
#vector_similarity_test = model.most_similar(test_final)

print(vector_test)
print("$$$$$$$$")
#print(similar_by_vector)
print(model.similar_by_vector(vector_test))
print("SimilarByVectorUPPP$$$$$$$$$")
#print(vector_similarity_test)
#	vectors_test.append(vector_test)
'''
#print(type(vectors_test))

#print(vector_test)
#MAYBE IF YOU USE PYTHON NORMALLY IT WORKS
#model.save("UnityDoc2Vec.model")
#print(model)

ranks =[]
second_ranks= []
sims = model.docvecs.most_similar([vector_test], topn=5)

#print(sims)
indexes = []
indexes_2 =[]
for sim in sims:
	indexes.append(sim[0])
	#print(model.docvecs[sim[0]])
#for sim in sims_2:
#	indexes_2.append(sim[0])
#print(indexes)
for item in indexes:
	print(dictionary[item])

##TODOOO,
##TRYING TO FIX THE BIGRAM PROBLEM
#Try gensim phrases
#Try bigram and input bigram to line
		#try the input where each word in bigram is added to test_input =test
#Try adding each word of bigram to line
		#try the input where each word in bigram is added to test_input 
#Default settings but duplicating each trained line
#Average inference vector
'''
