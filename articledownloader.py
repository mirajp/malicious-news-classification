import csv
from unidecode import unidecode
import newspaper
import re
import time
import articleDateExtractor

Article = newspaper.Article

credibleLabel = '0'
maliciousLabel = '1'


SRC_FILE = ?
OUTPUT_FILE = ?
#SRC_FILE_CLASS = maliciousLabel
#SRC_FILE_CLASS = credibleLabel
SRC_FILE_CLASS = ?
SRC_FILE_URL_INDEX = ?

articlesList = []
target = open(OUTPUT_FILE, 'w+')
target.write('label,URL,filepath,title,authors_attributed,num_characters,num_words,date\n')


with open(SRC_FILE, 'rt') as file:
    # Skip the column headers
    next(file)
    data_iter = csv.reader(file, delimiter = ',', quotechar = '"')
    articlesList = [data for data in data_iter]
print('Number of articles: ' + str(len(articlesList)))



def saveArticleContents(filepath, articleText):
    textfile = open(filepath, 'w')
    #articleText = unidecode(articleText.decode('utf-8'))
    articleText = unidecode(articleText)
    textfile.write(articleText + '\n')
    textfile.close
    return

def addArticle(featureClassList, prefix, articleLabel, articleURL, articleObj):
    articleTitle = articleObj.title
    authors = articleObj.authors
    articleText = articleObj.text
    
    articleTitle = re.sub(r'[\'\,\.\"\\\/\!\@\$\%\&\*]+', '', articleTitle)
    filename = prefix + '_' + re.sub(r'\W+', '', articleTitle)
    filepath = filename[:24] + '.txt'
    if (articleLabel == credibleLabel):
        filepath = './credible/' + 'r_' + filepath
    else:
        filepath = './malicious/' + 'f_' + filepath
    
    saveArticleContents(filepath, articleText)
    
    numChar = len(articleText)
    numWords = len(articleText.split())
    articleDate =  'NULL'
    d = articleDateExtractor.extractArticlePublishedDate(articleURL)
    if (type(d) == datetime.datetime):
        articleDate = d.date()
    
    featureClassList.write(articleLabel + ',' + articleURL + ',' + '"' + filepath + '"' + ',' + '"' + articleTitle + '"' + ',' + str(len(authors)) + ',' + str(numChar) + ',' + str(numWords) + ',' + articleDate + '\n')    
    return

for articleIter in range(0, len(articlesList)):
    articleURL = articlesList[articleIter][SRC_FILE_URL_INDEX]
    if 'youtube.com' not in articleURL and 'opinion' not in articleURL and 'radio' not in articleURL and 'blog' not in articleURL and 'video' not in articleURL and 'editorial' not in articleURL:
        #print(articleURL)
        if 'http://www.' not in articleURL:
            articleURL = 'http://www.' + articleURL
        
        if 'http://' not in articleURL:
            articleURL = 'http://' + articleURL
        
        prefix = str(articleIter + 1)
        try:
            articleObj = Article(url=articleURL, language='en')
            articleObj.download()
            articleObj.parse()
            if ('Page not found' not in articleObj.title):
                addArticle(target, prefix, SRC_FILE_CLASS, articleURL, articleObj)
        except newspaper.article.ArticleException:
            print("Error: " + prefix + "," + articleURL)

target.close