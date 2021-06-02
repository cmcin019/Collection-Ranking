# Import libraries 
import argparse
import os
import re
import string
import pandas as pd
import nltk
import itertools
import timeit
import torch

from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
from tld import get_tld
from top2vec import Top2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification

wn = nltk.WordNetLemmatizer()
nltk.download('wordnet')

class CollectionRank:
    '''
    '''
    def __init__(self, stopwords='stopwords.txt', corpus='collection.txt', queries='queries.txt', results='Results.txt', qc=-1, sc=False, bert='amberoad/bert-multilingual-passage-reranking-msmarco', feedBack=True, w2v=False):
        self.stopwords = stopwords
        self.corpus = corpus
        self.queries = queries
        self.results = results
        self.qc = qc
        self.sc = sc
        self.bert = bert
        self.feedBack = feedBack
        self.w2v = w2v
        self.dataset = None
        self.querieDataset = None
        self.stopwordLst = []
        pass

    @staticmethod
    def read_and_parse_stopwords(self):
        '''
        Reads from file and returns a list of stop words.
        \n\n\tParameters
        \n\t----------
        \n\string -> [stop words]
        '''

        # Open file from string 
        file = open(self.stopwords, "r")

        # Read from provided stopwords file
        raw_data_stopwords = file.read()

        # Assign list of stopwords
        self.stopwordLst = raw_data_stopwords.replace('\t', '\n').split('\n')

        file.close()

        pass

    @staticmethod
    def read_and_parse_queries(file):
        '''
        Reads from query file and returns a list of the tweet time stamps,
        a list of the tweet content, and a list of the tweet numbers.
        \n\n\tParameters
        \n\t----------
        \n\string -> [time stamps], [content], [numbers]
        '''

        # Read from provided query file
        raw_data_queries = file.read()

        # Split queries into a list
        tmpQriLst = raw_data_queries.replace('\t', '\n').split('\n')

        # List for dedicated query components
        querietitleLst = []
        querytweettimeLst = []
        querytweetnum = []

        for querie in tmpQriLst: # For each query from file

            if "<title>" in querie: # Title component 

                # Append title to designated list 
                querietitleLst.append(querie.replace(
                    "<title> ", "").replace(" </title>", ""))

                pass

            if "<querytweettime>" in querie: # Timestamp component 

                # Append timestamp to designated list 
                querytweettimeLst.append(querie.replace(
                    "<querytweettime> ", "").replace(" </querytweettime>", ""))

                pass

            if "<num>" in querie: # Number component 

                # Append number to designated list 
                querytweetnum.append(
                    int(querie.replace("<num> Number: MB0", "").replace(" </num>", "")))

                pass

            pass

        file.close()

        # Return lists -> query tweet timestamps, query tweet titles, query tweet numbers
        return querytweettimeLst, querietitleLst, querytweetnum

    @staticmethod
    def clean_text(self, txt):
        '''
        Retunrs a cleaned version of the input string.
        Removes punctuation and stopwords. 
        Removes hyperlinks while retaining pertinent information.
        Tokenizes text into a list of words. 
        Passes each word through a lemmatization algorithm.
        \n\n\tParameters
        \n\t----------
        \n\ttxt -> [words]
        '''

        # Extracting all URLs in order to retain pertinent information (i.e. sld and path)
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        urls = re.findall(regex, txt)
        for url in urls:
            if url is not None:
                try:
                    parsed = get_tld(url[0], fix_protocol=True,  as_object=True)
                    txt = txt.replace(
                        url[0], f'{parsed.domain} {parsed.parsed_url[2]}')
                except:
                    pass #ignore malformed urls that the regex found

        # Removing punctuation
        repNum = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        txt = txt.translate(repNum)

        # Tokenize the text
        tokens = re.split(r'\W+', txt)

        lst = []
        for word in tokens:

            if word not in self.stopwordLst:
                low = word.lower()
                lem = wn.lemmatize(low)

                lst.append(lem)
                pass
            pass

        return lst

    @staticmethod
    def createDataframe(self):
        '''
        Reads from tweet corpus file and returns a four column dataframe. 
        The first column contains the tweet numbers.
        The second colmn contains the original tweet content in string from. 
        The third column contians the content of the tweet after being run through the clean_text 
        function as a list of words.
        \n\n\tParameters
        \n\t----------
        \n\tfile -> [tweet numbers], [content in string from], [content in clean_text from]
        '''
        if os.name != 'posix':
            print('This script can only be run on posix compliant systems')
            exit()
        from pandarallel import pandarallel
        pandarallel.initialize()

        # Open file from string 
        file = open(self.corpus, "r", encoding="utf-8")

        self.dataset = pd.read_csv(file, '\t', header=0, names=['label', 'msg'], dtype=str)
        self.dataset.head()

        # Cleaning text - creating new column
        self.dataset['msg_clean'] = self.dataset['msg'].parallel_apply(lambda x: CollectionRank.clean_text(self, x))
        self.dataset.head()


        self.dataset.to_pickle('collection_cleaned.pickle')

        file.close()

        pass

    @staticmethod
    def createQuerieDataset(self, str):
        '''
        Read from query file and enters information into a four column dataframe. 
        The first column contains the query labels.
        The second coulmn contains the query content as a string. 
        The third contains the query numbers. 
        The last column contains the the content of the query after being run 
        through the clean_text function as a list of words.
        \n\n\tParameters
        \n\t----------
        \n\tfile -> [query labels], [content in string from], [query numbers], [content in clean_text from]
        '''
        if os.name != 'posix':
            print('This script can only be run on posix compliant systems')
            exit()
        from pandarallel import pandarallel
        pandarallel.initialize()

        # Open file from string 
        file = open(str, "r")

        querytweettimeLst, querietitleLst, querytweetnum = CollectionRank.read_and_parse_queries(file)
        self.querieDataset = pd.DataFrame({
            'label': querytweettimeLst,
            'queries': querietitleLst,
            'number': querytweetnum
        })
        self.querieDataset.head()

        # Cleaning text - creating new column
        self.querieDataset['queries_clean'] = self.querieDataset['queries'].parallel_apply(
            lambda x: CollectionRank.clean_text(self, x))
        self.querieDataset.head()

        self.querieDataset.to_pickle('queries_cleaned.pickle')

        file.close()
        
        pass

    # will find synonyms for pairs of words (Gives better results)
    @staticmethod
    def addSynonyms_pair_of_words(querie, model):
        '''
        \n\n\tParameters
        \n\t----------
        '''
        newQuerie = querie
        for word1, word2 in itertools.combinations(querie, 2):
            if(word1!=word2):
                similarity = model.wv.similarity(word1, word2)  
                if(similarity>=0.9): # checks if the similarity between the two words is atleast 90%
                    synonym = model.wv.most_similar(positive=[word1,word2])[0][0] #adds the synonym ranked the most relevant between the two words 
                    if(synonym not in newQuerie):
                        newQuerie.append(synonym)
        return newQuerie

    @staticmethod
    def rankDocs(self, bm25, testQuerie):
        '''
        Returns a sorted list, by value, of dictionaries.
        The key is a tweet id, and the value is the score of the given tweet 
        after being calculated using bm25 against a query.
        \n\n\tParameters
        \n\t----------
        \n\tBM25Okapi, query label, dataframe -> [ {id, score} ]    
        '''
        doc_scores = bm25.get_scores(testQuerie)
        x=0
        dictSum = {}
        for sim in doc_scores:
            dictSum[self.dataset.index[x]] = sim
            x += 1

        rankedDocs = [(k, v) for k, v in sorted(
            dictSum.items(), key=lambda item: item[1], reverse=True)]

        return rankedDocs

    @staticmethod
    def bertCall(self, rankedDocs, tokenizer, query, device, pt_model):
        '''
        Re ranks up to 1000 documents using a pretrained model. 
        Returns a new list of ranked documents, sorted by the rank
        \n\n\tParameters
        \n\t----------
        \n\trankedDocs, dataset, tokenizer, query, device, pt_model - > [ (id, rank) ]
        '''
        newRankedDocs = []
        for rankedDoc in range(min(1000, len(rankedDocs))):

            rankedDocID = rankedDocs[rankedDoc][0]

            tmp_doc = self.dataset.loc[[rankedDocID], ['msg']]['msg'][0]

            pt_batch = (tokenizer.encode_plus(
                    query,
                    tmp_doc,
                    return_tensors="pt"
            ))

            # Load batch on device (this might be redundant if already using CPU)
            # but it is required when using gpu
            pt_batch.to(device)

            pt_outputs = pt_model(**pt_batch)[0]

            pt_predictions = torch.softmax(pt_outputs, dim=1).tolist()[0][1]
            newRankedDocs.append((rankedDocs[rankedDoc][0], pt_predictions))

            pass

        newRankedDocs.sort(key=lambda a: a[1], reverse=True)
        return newRankedDocs

    @staticmethod
    def top2vecProcess(self, model, query_num):
        query = self.querieDataset['queries'][query_num]
        model.add_documents(documents=[query], doc_ids=[str(query_num)])

        _, document_scores, document_ids = model.search_documents_by_documents(doc_ids=[str(query_num)], num_docs=1000)

        rankedDocs = list(map(lambda x, y:(x,y), document_ids, document_scores))

        model.delete_documents(doc_ids=[str(query_num)])

        return rankedDocs


    @staticmethod
    def preProcess(self):
        CollectionRank.read_and_parse_stopwords(self)
        if self.sc:
            self.dataset = pd.read_pickle('collection_cleaned.pickle')
        else:
            CollectionRank.createDataframe(self)
        self.dataset.set_index("label", inplace = True) 

        if self.sc:
            self.querieDataset = pd.read_pickle('queries_cleaned.pickle')
        else:
            CollectionRank.createQuerieDataset(self, self.queries)

        # Open file from string 
        file = open(self.results, "w+")

        query_count = self.qc if self.qc != -1 else self.querieDataset.count()['queries']

        bm25 = BM25Okapi(self.dataset['msg_clean'])

        if self.w2v != False:
            # train and create word2vec model using CBOW
            model_CBOW = Word2Vec(self.dataset["msg_clean"].tolist())
            return file, query_count, model_CBOW, bm25

        return file, query_count, None, bm25

    @staticmethod
    def relevanceFeedback(self, rankedDocs=None, testQuerie=None, bm25=None, tokenizer=None, query=None, device=None, pt_model=None, query_num=None, a=None, b=None, msg=None):

        itr = 0
        while (True):
            if itr > 5:
                break
            _, tweetRank = rankedDocs[itr]
            if tweetRank > 0.65 or not (itr > 4 or tweetRank < 0.5):
                if msg == 'msg_clean':
                    testQuerie.extend(self.dataset.loc[[rankedDocs[itr][0]], [msg]][msg][0])
                    pass
                else:
                    query = query + " " + self.dataset.loc[[rankedDocs[itr][0]], [msg]][msg][0]
                    pass

                itr += 1
            else:
                break
            pass

        if msg == 'msg_clean':
            rankedDocs = CollectionRank.rankDocs(self, bm25, testQuerie)
            pass
        elif tokenizer == None :
            rankedDocs = CollectionRank.top2vecProcess(self, pt_model, query_num)
            pass
        else:
            rankedDocs = CollectionRank.bertCall(self, rankedDocs, tokenizer, query, device, pt_model)
            pass


        return rankedDocs

    @staticmethod
    def bm25Process(self, setSize, model_CBOW, bm25):

        print("Now working on : " + self.querieDataset['queries'][setSize] + " : " + str(setSize+1) + "\n")

        testQuerie = self.querieDataset['queries_clean'][setSize]

        if self.w2v != False :
            testQuerie = CollectionRank.addSynonyms_pair_of_words(testQuerie, model_CBOW)
            pass
        rankedDocs = CollectionRank.rankDocs(self, bm25, testQuerie)

        if self.feedBack == True:
            rankedDocs = CollectionRank.relevanceFeedback(self, rankedDocs=rankedDocs, testQuerie=testQuerie, bm25=bm25, a=18, b=10, msg='msg_clean')
            
            pass
        
        return rankedDocs

    @staticmethod
    def results(self, setSize, rankedDocs, file):
        testQuerieNum = self.querieDataset['number'][setSize]
        # We only want the top 1000 results or less
        for x in range(min(1000, len(rankedDocs))):
            rank = str(x + 1)
            tweetId, tweetRank = rankedDocs[x]
            file.write(str(testQuerieNum) + "\tQ0\t" + tweetId +
                    "\t" + rank + "\t" + str(tweetRank) + "\tmyRun\n")
            pass
        pass 


    def bm25(self):
        
        file, query_count, model_CBOW, bm25 = CollectionRank.preProcess(self)
        
        for setSize in range(query_count):
        # for setSize in range(1):
            rankedDocs = CollectionRank.bm25Process(self, setSize, model_CBOW, bm25)
            CollectionRank.results(self, setSize, rankedDocs, file)
            pass

        file.close()

        pass


    def Bert(self):

        model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
        print("(Down)loading pretrained model")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pt_model.to(device)
        
        file, query_count, model_CBOW, bm25 = CollectionRank.preProcess(self)
        
        for query_num in range(query_count):
        # for query_num in range(1):
            rankedDocs = CollectionRank.bm25Process(self, query_num, model_CBOW, bm25)

            query = self.querieDataset['queries'][query_num]
            rankedDocs = CollectionRank.bertCall(self, rankedDocs, tokenizer, query, device, pt_model)

            if self.feedBack == True:
                rankedDocs = CollectionRank.relevanceFeedback(self, rankedDocs=rankedDocs, bm25=bm25, tokenizer=tokenizer, query=query, device=device, pt_model=pt_model, a=0.99945, b=0.012, msg='msg')
                pass

            CollectionRank.results(self, query_num, rankedDocs, file)
            pass

        file.close()

        pass

    def top2vec(self):

        file, query_count, model_CBOW, bm25 = CollectionRank.preProcess(self)
        
        model = Top2Vec(documents=self.dataset['msg'].tolist(), speed="learn", workers=8, document_ids=self.dataset.index.values.tolist(), embedding_model='universal-sentence-encoder')

        for query_num in range(query_count):
        # for query_num in range(1):

            print("Now working on : " + self.querieDataset['queries'][query_num] + " : " + str(query_num+1) + "\n")

            query = self.querieDataset['queries'][query_num]

            rankedDocs = CollectionRank.top2vecProcess(self, model, query_num)

            if self.feedBack == True:
                rankedDocs = CollectionRank.relevanceFeedback(self, rankedDocs=rankedDocs, pt_model=model, query=query, query_num=query_num, a=0.5, b=0.4, msg='msg')
                pass

            CollectionRank.results(self, query_num, rankedDocs, file)
 
            pass

        file.close()


        pass



    # End of class
    pass













def main():

    cr = CollectionRank(feedBack=True)
    cr.Bert()
    print("End of program")

    pass
















# Call main function
if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
