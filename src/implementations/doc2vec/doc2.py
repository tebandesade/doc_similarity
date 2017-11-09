from gensim.models import Doc2Vec
import numpy as np
from gensim.parsing import preprocessing as pre
from gensim.models import Phrases as ph
from gensim.models.phrases import Phraser
import os
import collections
import smart_open
import random
import gensim
import nltk
from nltk.corpus import stopwords
import string
from nltk.util import ngrams
'''

path_ = "../out_clean"
##Double preprocessing
def create_bigrams(fname):
	lines = []
	with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
		for line in f:
			line = line.strip()
			line = pre.strip_tags(line)
			line = pre.strip_punctuation(line)
			line = line.strip()
			line_token = nltk.word_tokenize(line)
			lines.append(line_token)
	text = [j for t in lines for j in t]
	sentences.append(text)

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
			line_token = nltk.word_tokenize(line)
			#bigrams  = ngrams(line_token,2)

			#lines.append(line.split())
			lines.append(line_token)
			#doc.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line),[i]))
	text = [j for t in lines for j in t]
	sentences.append(text)
	return gensim.models.doc2vec.TaggedDocument(text,[l])
	#documents.append(doc)

def read_sentence(sentence,i):
	#print(bigram[sentence])
	return gensim.models.doc2vec.TaggedDocument(bigram[sentence],[i])

files     = os.listdir(path_)
i = 0
documents = []	
dictionary = {} 
sentences  = []
contador = 0
for file_ in files:
	dictionary[contador] = file_
	file_path = os.path.join(path_,file_)
	create_bigrams(file_path)
	contador = contador +1 

frases = ph(sentences)
bigram = Phraser(frases)
bigram.save('bigrams_40')
for sent in sentences:
	doc_2v = read_sentence(sent,i)
	i = i +1
	documents.append(doc_2v)

#print(documents[0])


#for file_ in files:
#	file_path = os.path.join(path_,file_)
#	doc_2v = #read_corpus_(file_path,i)
#	dictionary[i] = file_
#	i=i+1
#	documents.append(doc_2v)




model   = Doc2Vec(documents,size=150,hs=1,min_count=2,workers=4, iter=20,dbow_words=1,dm=0)
'''
dictionary = np.load("dic40.npy")
model = Doc2Vec.load('ricardo_col40')
bigram_ = Phraser.load('bigrams_40')
dic_mapping = np.load('dic_mapping.npy')

stop = stopwords.words('english') + list(string.punctuation)
input_ = 'We are trying to use Host Migration with online matchmaker.To make it simple now we are using it now with "Show GUI".Basically we added a custom NetworkMigrationManager, where we only overrided OnClientDisconnectedFromHost, where we call the base function and set a flag to disable any message sending after migration (for testing). See the attached file: HostMigration.cs. We start with 3 players, then the server quits, and host migration happens between the 2 remaining machines by using the UI buttons. It seems like that it happens successfully, there will be a new server, and the another client receives this log: NetworkClient Reconnect::ffff:52.28.11.218:5054 UnityEngine.Networking.NetworkMigrationManager:OnGUI(). But when we try to send the first message (through a chat), we get this error: Send command attempted with no client running [client=hostId: 0 connectionId: 1 isReady: False channel count: 2].UnityEngine.Networking.NetworkBehaviour:SendCommandInternal(NetworkWriter, Int32, String) NetworkPlayer:CallCmdServerChatMessage(PlayerId, String) This is the point where we are stuck...We received this Send command attempted with no client running... message all the time when we try to send any message. What could be the problem? '
#input_  = "I've asked repeatedly about this and given often completely incorrect answers from supposed developers. Its both absurd that this wasnt done years ago, and hasnt been done in 5.4 with the editor now supporting retina. Basically, i'm holding off telling users they need to upgrade from osx 10.7 right now as i'm using Unity 5.2, as 5.3/5.4 has zero additional benefit. Lack of Retina is a deal breaker for me and the reason i won't be using Unity in any future projects ore recommending it to anyone. "
#
#input_ = 'Is there any reason why the same exact scene, with a large realtime spot on the play area, would have much more pixelated hard shadows under Fantastic quality with Very High Resolution shadows when using 5.4.3f1 instead of 5.3.7f1?'
#input_  ='Help! Earlier this year, Allegorithmic released Substance Designer 6 (along with Substance Painter 2.5), which added some great new features and enhanced the functionality of their Substance .sbsar files. Unfortunately, these do not appear to work properly in Unity. Aside from Substances created in Substance Designer 6x just not loading, I find that .sbsar files in Unity can be rendered at no higher a resolution than 2048x2048, despite Allegorithmic\'s format supporting higher resolutions. Checking the software manufacturer\'s forums, they say that unfortunately this is entirely in the hands of Unity. So I\'m posting here and asking, when can we hope to have full compatibility and feature support for Allegorithmic .sbsar files?'


test = input_.lower()
test = pre.strip_punctuation(test)
test = pre.strip_tags(test)
test = pre.strip_numeric(test)
test_final = [i for i in nltk.word_tokenize(test.encode('utf-8')) if i not in stop]
#bigrams  = ngrams(test_final,2)
result_test_bigram = bigram_[test_final]
#print (result_test_bigram)
list_input = []
#for bi in bigrams:
#	for word in bi:
#		list_input.append(word)
#print(list_input)
#for i in range(10):
#print(help(model))
print(result_test_bigram )
#test_vector = model[test_final]
vector_test = model.infer_vector(result_test_bigram,steps=10000)
#print(model.docvecs[test_final])
#similar_by_vector = model.similar_by_vector(vector_test)
#vector_similarity_test = model.most_similar(test_final)

#print(vector_test)
#print("$$$$$$$$")
#print(similar_by_vector)
#print(model.similar_by_vector(vector_test))
#print("SimilarByVectorUPPP$$$$$$$$$")
#print(vector_similarity_test)
#	vectors_test.append(vector_test)

#print(type(vectors_test))

#print(vector_test)
#MAYBE IF YOU USE PYTHON NORMALLY IT WORKS
#model.save("UnityDoc2Vec.model")
#print(model)

ranks =[]
second_ranks= []
print(' '.join(test_final))
print(model.score([test_final]))
#HAVE TO CREATE A THRESHOLD FOR CONFIDENCE SCORE
#Proposing 0.34
sims = model.docvecs.most_similar([vector_test], topn=10)
print(sims)
#print(sims)
indexes = []
indexes_2 =[]

for sim in sims:
	indexes.append(sim[0])
	#print(model.docvecs[sim[0]])
#for sim in sims_2:
#	indexes_2.append(sim[0])
#print(indexes)
#print(dic_mapping)

lista_elem = []
#print dic_mapping
for element in indexes:
	name_ = dictionary.item()[element]

	print('webpage:',name_)
	try:
		print(dic_mapping.item()[name_][0])
		#print(dic_mapping.item()[name_][1])
	except:
		print('error with', name_)
'''
	
	if result_temp[0] in mapping_toarea:
		print (mapping_toarea[result_temp[0]])
	else:
		print(result_temp[0])
	#lista_elem.append(result_temp[0])
	#for el in result_temp:
	#	if el in mapping_toarea:
	#		print(mapping_toarea[el])
	#else:
	#	print(el)
	#	#lista_elem.append(el)

print(set(lista_elem))
#print(dic_mapping.item()['UnityIAPTizen.md'])

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
