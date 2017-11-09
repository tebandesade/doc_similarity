# doc_similarity
A Q/A, doc retriever 

This project aims to develop a program that's able to process some text (docs) and create a probabilistic model out of it. 

Once it is set,  the program is able to accept new input (text) as a query and get the k most documents that are related to the query. (Doc retriever)

Description:
For the project two approaches are implemented (intend to do more). One is an implementation done using gensims doc2vec model ("AI"). The other implementation is a TFIDF model (similar to Facebook's first part of DQRA). 

Note: It is not organized; it is in development (debug phase) and can't show something directly. If you want to see something, in the doc2vec add a path of files to path_, and change the input_ text and run the python doc2 script.

If you want to test out the TF-IDF implementation.
First, create a list of files (docs) and store them in a directory.
Second, run create_token_dictionary and pass as an argument the name of the directory you created.
*Directory has to be in the same directory of the python script 
Third, run query_predction and as argument pass a text-query    

The data used for this will be Unity's documentation
TODO
Develop a Q/A agent
