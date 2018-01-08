import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Embedding, Activation
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report, accuracy_score
from NewsArticleDataset import NewsArticleDataset

GLOVE_DIR = "./glove.6b/"
GLOVE_DIR = GLOVE_DIR.replace("./", sys.path[0] + "/")
GLOVE_TEXTFILE = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")

MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
TRAINABLE = False

NUM_UNITS = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 10
DROPOUT_RATE = 0.5

lossFn = "binary_crossentropy"
optFn = "adam"
actFn = "sigmoid"

seed = 7
np.random.seed(seed)

bestEpochs = 1
bestBatchSize = 32

# Load the words features
print("Loading the training, testing, validation datasets...")
ArticleDataset = NewsArticleDataset("./credible_news_v5.csv", "./malicious_news_v5.csv")

# header = [label, #, URL, filepath, title, authors_attributed, num_characters, num_words, date, google_sentiment_score, google_sentiment_magnitude, msft_sentiment_score]
header = ArticleDataset.getHeader()
(trainFeatures, trainLabels, testFeatures, testLabels, tuneFeatures, tuneLabels) = ArticleDataset.getTrainTuneTestSets("10/13/2016", VALIDATION_SPLIT)

trainLabels = (trainLabels==1).astype(np.int32)
testLabels = (testLabels==1).astype(np.int32)
tuneLabels = (tuneLabels==1).astype(np.int32)

textfileIndex = header.index("filepath")

# Read all of the articles into memory
def loadDatasetArticles(dataset, fileIndex):
    filesList = dataset[:,fileIndex]
    textfiles = [open(f) for f in filesList]
    texts = [f.read() for f in textfiles]
    for f in textfiles:
        f.close
    return texts

print("Loading the articles...")
trainingTexts = loadDatasetArticles(trainFeatures, textfileIndex)
testingTexts = loadDatasetArticles(testFeatures, textfileIndex)
validationTexts = loadDatasetArticles(tuneFeatures, textfileIndex)
print("Found %d training texts, %d testing texts, %d validation texts" % (len(trainingTexts), len(testingTexts), len(validationTexts)))


# Vectorize the text samples into a 2D integer tensor
print("Fitting tokenizer on training articles...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainingTexts)
word_index = tokenizer.word_index

trainData = pad_sequences(tokenizer.texts_to_sequences(trainingTexts), maxlen=MAX_SEQUENCE_LENGTH)
testData = pad_sequences(tokenizer.texts_to_sequences(testingTexts), maxlen=MAX_SEQUENCE_LENGTH)
validationData = pad_sequences(tokenizer.texts_to_sequences(validationTexts), maxlen=MAX_SEQUENCE_LENGTH)


# Build a mapping of words to their word embedding vector 
print("Loading and indexing word vectors...")
embeddings_index = {}
f = open(GLOVE_TEXTFILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors!" % len(embeddings_index))


# Prepare embedding matrix
print("Preparing word embedding matrix...")
num_words = min(MAX_NB_WORDS, len(word_index))+1
print("num_words (input_dim) = " + str(num_words))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Load pre-trained word embeddings into an Embedding layer
# trainable = False --> keep the embeddings fixed
embeddingLayer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=TRAINABLE)

def createModel():
    global MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_UNITS, embeddingLayer
    global lossFn, optFn, actFn, DROPOUT_RATE
    model = Sequential()
    
    if (embeddingLayer == None):
        embeddingLayer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
    
    model.add(embeddingLayer)
    #model.add(LSTM(NUM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))
    model.add(LSTM(NUM_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation=actFn)) # condense the output layer (size = NUM_UNITS) to provide binary classification
    model.compile(loss=lossFn, optimizer=optFn, metrics=['accuracy'])
    return model


batch_size = [24]
epochs = [8]
#verbose = [0] # Silence the grid search
verbose = [1] # Show bar graphs of progress from the grid search output during each epoch

param_grid = dict(batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=[(validationData, tuneLabels)])
model = KerasClassifier(build_fn=createModel)
clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

print("Performing grid search...")
grid_result = clf.fit(trainData, trainLabels)
print("Best: %f using batch_size = %.1f, epochs = %.1f" % (grid_result.best_score_, grid_result.best_params_["batch_size"], grid_result.best_params_["epochs"]))
bestEpochs = grid_result.best_params_["epochs"]
bestBatchSize = grid_result.best_params_["batch_size"]

print("Training model with 'best params': num_epochs = " + str(bestEpochs) + ", batch_size = " + str(bestBatchSize) + ", NUM_UNITS = " + str(NUM_UNITS) + ", dropout = " + str(DROPOUT_RATE))
model = createModel()
model.fit(trainData, trainLabels, batch_size=bestBatchSize, epochs=bestEpochs, validation_data=(validationData, tuneLabels))

print("Evaluating model...")
scores = model.evaluate(testData, testLabels, verbose=0)
predictions = model.predict(testData, verbose=0)
predictions = np.round(predictions)
acc = accuracy_score(testLabels, predictions)

print("Using num_epochs = " + str(bestEpochs) + ", batch_size = " + str(bestBatchSize) + ", NUM_UNITS = " + str(NUM_UNITS) + ", dropout = " + str(DROPOUT_RATE)) 

for metricIter in xrange(0, len(scores)):
    metricName = model.metrics_names[metricIter]
    metricValue = scores[metricIter]
    print("Models's " + metricName + " = " + str(metricValue))
    
target_names = ["credible", "malicious"]

print("Confusion matrix:")
print(classification_report(testLabels, predictions, target_names=target_names))
