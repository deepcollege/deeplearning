# Word2vec

References:
1.  https://medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1
2.  https://medium.com/deep-math-machine-learning-ai/chapter-9-2-nlp-code-for-word2vec-neural-network-tensorflow-544db99f5334


#### 1. Corpus Vectorisation (preprocessing)

We can use counter vectorisation but we might get high count vectors for one document
and low count vectors for others. Intead we will favour TF-IDF(Term frequency-Inverse term frequency).

tf-idf is a weighting factor which is used to get the important features from the documents(corpus).

It actually tells us how important a word is to a document in a corpus, the importance of a word increases proportionally to the number of times the word appears in the individual document, this is called Term Frequency(TF).

Ex : document 1:

“ Mady loves programming. He programs all day, he will be a world class programmer one day ”

if we apply tokenization, steeming and stopwords (we discussed in the last story) to this document, we get features with high count like → program(3), day(2),love(1) and etc….

***TF*** = (no of times the word appear in the doc) / (total no of words in the doc)

Here program is the highest frequent term in the document.

so program is a good feature if we consider TF.

However, if multiple documents contain the word “program” many times then we might say…

it’s also a frequent word in all other documents in our corpus so it does not give much meaning so it probably may not be an important feature.

To adjust this we use IDF.

The inverse document frequency is a measure of how much information the word provides, that is, whether the term is common or rare across all documents.

***IDF*** — Log(total no of documents / no of documents with the term t in it).

so TF-IDF = TF * IDF.

Problems:
1. TF-IDF or CounterVectorisations do not maintain order or semantic relationship
between the words

2. Instead we need to build Word2Vec model -> Converts this high dimensional vector (10000 sized) into low dimensional vector (let’s say 200 sized)


#### 2. Word 2 vec

Word2vec takes care of 2 things:

1. Converts this high dimensional vector (10000 sized) into low dimensional vector (let’s say 200 sized)
2. Maintains the word context (meaning)

the word context / meaning can be created using 2 simple algorithms which are

1. Continuous Bag-of-Words model (CBOW)

Ex: Text= “Mady goes crazy about machine leaning” and window size is 3

-> [ [“Mady”,”crazy” ] , “goes”] → “goes” is the target word


2. Skip-Gram model

It takes one word as input and try to predict the surrounding (neighboring) words,

[“Mady”, “goes”],[“goes”,”crazy”] → “goes” is the input word and “Mady” and “Crazy” are the surrounding words (Output probabilities)


What is word2vec in short? 

→ it’s a neural network training for all the words in our dictionary to get the weights(vectors )

→ it has word embeddings for every word in the dictionary


