import ebooklib, re, os, gensim
from ebooklib import epub
from PyPDF2 import PdfReader

import numpy as np

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords, names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import TruncatedSVD # also known as Latent semantic analysis

import enchant

ENGLISH_DICT1 = enchant.Dict("en_UK")
ENGLISH_DICT2 = enchant.Dict("en_US")

STOP_WORDS = stopwords.words("english")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

def is_english_word(word):
    # Initialize the Enchant English dictionary
    return (ENGLISH_DICT1.check(word) or ENGLISH_DICT2.check(word))

def preprocess(paragraphs):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    processed_docs = []
    
    for paragraph in paragraphs:
        words = gensim.utils.simple_preprocess(paragraph, min_len = 3, deacc=True)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        filtered_words = [word for word in lemmatized_words if ((word not in stop_words)and(is_english_word(word)))]
        #stemmed_words = [stemmer.stem(word) for word in filtered_words]
        #processed_doc = " ".join(stemmed_words)
        processed_doc = " ".join(filtered_words)
        processed_docs.append(processed_doc)
    return processed_docs

def merge_strings_until_limit(strings, min_length, max_length, test_for_max = 0):
    merged_string = ""
    merged_strings = []
    
    for s in strings:
        if len(merged_string) <= min_length:
            merged_string += s
        
        elif len(merged_string) > max_length and test_for_max<5:
                splitParagraph = merged_string.split('.')
                splitParagraphRePoint = []
                for sp in splitParagraph:
                    splitParagraphRePoint.append(sp+'.')
                
                merged = merge_strings_until_limit(splitParagraphRePoint, min_length, max_length, test_for_max+1)
                merged_strings.extend(merged)
                merged_string = s
        else:
            merged_strings.append(merged_string)
            merged_string = s
    
    if merged_string:
        merged_strings.append(merged_string)
    
    return merged_strings

def read_epub_paragraphs(epub_file, ID, filetype):
    book = epub.read_epub(epub_file)
    paragraphs = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode('utf-8')
        content = re.sub('<[^<]+?>', '', content)
        content = re.sub('\s+', ' ', content)
        content = re.sub('\n', ' ', content)
        
        paragraphs.extend(content.strip().split("&#13;"))
    
    paragraphs = merge_strings_until_limit(paragraphs, 200, 1000)
    paragraphs = [{'paragraph':paragraphs[i], 'nr':i, 'ID':ID, 'type':filetype} for i in range(len(paragraphs))]

    return paragraphs[1:-1]

def read_pdf_paragraphs(pdf_file, ID, filetype):
    paragraphs = []
    with open(pdf_file, 'rb') as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text = page.extract_text()
            text = re.sub('<[^<]+?>', '', text)
            text = re.sub('\s+', ' ', text)
            text = re.sub('\n', ' ', text)
            paragraphs.extend(text.strip().split("&#13;"))
            
    paragraphs = merge_strings_until_limit(paragraphs, 200, 1000)
    paragraphs = [{'paragraph':paragraphs[i], 'nr':i, 'ID':ID, 'type':filetype} for i in range(len(paragraphs))]
    return paragraphs[1:-1]



class TextModel:
    def __init__(self, files, vectorization='lsa', dimension=200, min_df=2):
        self.vectorization = vectorization
        self.paragraphs = []

        for f in files:
            ID = os.path.splitext(os.path.basename(f))[0]
            filetype = f.split('.')[-1]
            if filetype == 'epub':
                paragraph = read_epub_paragraphs(f, ID, 'epub')
            elif filetype == 'pdf':
                paragraph = read_pdf_paragraphs(f, ID, 'pdf')
            self.paragraphs.extend(paragraph)

        self.preprocessed_paragraphs = preprocess(p['paragraph'] for p in self.paragraphs)

        if self.vectorization == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.vector_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
        elif self.vectorization == 'lsa':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
            self.svd = TruncatedSVD(n_components=dimension, algorithm='randomized')
            self.vector_matrix = self.svd.fit_transform(self.tfidf_matrix)

        self.nnModel = NearestNeighbors(n_neighbors=10,
                                        metric='cosine',
                                        algorithm='brute',
                                        n_jobs=-1)
        self.nnModel.fit(self.vector_matrix)
        
    def vectorize(self, query):
        if self.vectorization == 'lsa':
            processedQuery = preprocess([query])[0]
            tfidf_query = self.tfidf_vectorizer.transform([processedQuery])
            query_vector = self.svd.transform(tfidf_query)
            return query_vector
        elif self.vectorization == 'tfidf':
            processedQuery = preprocess([query])[0]
            query_vector = self.tfidf_vectorizer.transform([processedQuery])
            return query_vector

    def search(self, query, n=3, distance=False):
        qv = self.vectorize(query)
        neighbours = self.nnModel.kneighbors(qv, n, return_distance=distance)[0]
        paragraphs = [tuple(p.items()) for p in (self.paragraphs[i] for i in neighbours)]
        unique_paragraphs = list(set(paragraphs))
        return [dict(p) for p in unique_paragraphs]

    def get_key_words(self, v, n=10):
        if self.vectorization == 'lsa':
            v = self.svd.inverse_transform(v)[0]
            top_indices = np.argpartition(v, -n)[-n:]
            words = self.tfidf_vectorizer.get_feature_names_out()
            return [words[i] for i in top_indices]
        elif self.vectorization == 'tfidf':
            top_indices = np.argpartition(v, -n)[-n:]
            words = self.tfidf_vectorizer.get_feature_names_out()
            return [words[i] for i in top_indices]







