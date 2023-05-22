import requests
from bs4 import BeautifulSoup #parsing HTML/XML documents
from nltk.corpus import stopwords # Natural Language Toolkit
from nltk.stem import SnowballStemmer# that reduces words to their root form or base form
import string #provides a collection of useful string constants and functions.
from sklearn.feature_extraction.text import TfidfVectorizer
#transforms a collection of raw text documents to a matrix of TF-IDF features.
from sklearn.cluster import KMeans# algorithm used for clustering data.

def get_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text()
    return content

def preprocess_text(text):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def cluster_web_pages(urls, num_clusters=2):
    pages = []
    for url in urls:
        page = get_page_content(url)
        if page:
            pages.append(page)
    pages = [preprocess_text(page) for page in pages]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(pages)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)


    for i in range(num_clusters):
        print(f'Cluster {i}:')
        for j in range(len(pages)):
            if kmeans.labels_[j] == i:
                print(urls[j])
        print()


# Example usage
urls = []
num_urls = int(input('Enter the number of URLs you want to cluster: '))
for i in range(num_urls):
    url = input(f'Enter URL {i+1}: ')
    urls.append(url)
num_clusters = int(input('Enter the number of clusters: '))
cluster_web_pages(urls, num_clusters)
