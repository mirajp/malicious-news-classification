import csv
import numpy as np

class NewsArticleDataset(object):
    #Initialization loads the two classes (credible (0) and malicious (1)) of articles into memory
    def __init__(self, credibleCSVPath, maliciousCSVPath):
        self.credibleCSV = credibleCSVPath      # Class 0
        self.maliciousCSV = maliciousCSVPath    # Class 1
        self.numCredible = 0
        self.numMalicious = 0
        
        with open(credibleCSVPath, 'rt') as file:
            reader = csv.reader(file, delimiter = ',', quotechar = '"')
            self.header = reader.next()
            self.credibleFeatures = [data for data in reader]
        self.credibleFeatures = np.array(self.credibleFeatures)
        self.numCredible = len(self.credibleFeatures)
        self.credibleLabels = np.zeros((self.numCredible,), dtype=np.int)

        with open(maliciousCSVPath, 'rt') as file:
            reader = csv.reader(file, delimiter = ',', quotechar = '"')
            header = reader.next()
            self.maliciousFeatures = [data for data in reader]
        self.maliciousFeatures = np.array(self.maliciousFeatures)
        self.numMalicious = len(self.maliciousFeatures)
        self.maliciousLabels = np.ones((self.numMalicious,), dtype=np.int)
        
        # Try to ensure the two CSVs have the same type of information
        if len(header) != len(self.header):
            raise ValueError("Header Mismatch: The two CSV's headers' lengths don't match!")
        
        for h in xrange(0, len(header)):
            if header[h] != self.header[h]: # If the header of malicious CSV doesn't match header of credible CSV
                raise ValueError("Column Mismatch: The " + str(h+1) + "th column in the malicious CSV is '" + header[h] + "', but '" + self.header[h] + "' in the credible CSV!")
        return
    

    # Returns the header for each class's dataset
    def getHeader(self):
        return self.header
    
    
    # Returns features for class 0 and class 1, separately
    def getCorpus(self):
        return (self.credibleFeatures, self.maliciousFeatures)
    
    
    def getTrainTestSets(self, splitValue, splitColumn="date"):
        return self.getTrainTuneTestSets(splitValue, 0.0, splitColumn)
    
    
    """
    Assumes the data in splitColumn is already sorted, so every article with a value before splitValue is put
    into the training set (or tuning), and every article with a value after splitValue is put in the testing set
    Note: the splitValue must be found for at least one article in each class
    """
    def getClassTrainTestSets(self, articleClass, splitValue, splitColumn="date"):
        if splitColumn not in self.header:
            raise ValueError("splitColumn '" + splitColumn + "' was not found in the header")

        classFeatures = None
        classLabels = None
        className = None
        if isinstance(articleClass, basestring):
            articleClass = articleClass.lower()
            if articleClass == "malicious":
                classFeatures = self.maliciousFeatures
                classLabels = self.maliciousLabels
            elif articleClass == "credible":
                classFeatures = self.credibleFeatures
                classLabels = self.credibleLabels
            else:
                raise ValueError("articleClass must either be 'credible' or 'malicious'")
        elif isinstance(articleClass, (int, long)):
            if articleClass == 1:
                classFeatures = self.maliciousFeatures
                classLabels = self.maliciousLabels
                articleClass = "malicious"
            elif articleClass == 0:
                classFeatures = self.credibleFeatures
                classLabels = self.credibleLabels
                articleClass = "credible"
            else:
                raise ValueError("articleClass must either be 0 for credible or 1 for malicious")
        else:
            raise ValueError("articleClass must either be an int (0, 1) or a str ('credible', 'malicious')")
        
        # Extract the articles' splitColumn
        try:
            splitColumnIndex = self.header.index(splitColumn)
        except ValueError:
            raise ValueError("Class '" + articleClass + "' does not contain the feature column '" + splitColumn + "'")
        classArticleSplitValues = classFeatures[:,splitColumnIndex].tolist()

        # Use the splitValue to split up the class into training and testing
        try:
            classTrainingEnd = classArticleSplitValues.index(splitValue)
        except ValueError:
            raise ValueError("Class '" + articleClass + "' does not contain the value '" + splitValue + "' in the '" + splitColumn + "' feature column")

        classTrainFeatures = classFeatures[0:classTrainingEnd]
        classTrainLabels = classLabels[0:classTrainingEnd]
        classTestFeatures = classFeatures[classTrainingEnd:]
        classTestLabels = classLabels[classTrainingEnd:]

        return (classTrainFeatures, classTrainLabels, classTestFeatures, classTestLabels)
    

    def getTrainTuneTestSets(self, splitValue, tuningProportion=0.0, splitColumn="date"):
        credibleTrainFeatures, credibleTrainLabels, credibleTestFeatures, credibleTestLabels = self.getClassTrainTestSets(0, splitValue, splitColumn)
        maliciousTrainFeatures, maliciousTrainLabels, maliciousTestFeatures, maliciousTestLabels = self.getClassTrainTestSets(1, splitValue, splitColumn)

        tuneFeatures = np.array([])
        tuneLabels = np.array([])
        
        # If size for tuning/validation set provided, extract that proportion
        # from the (chronologically) later articles in each class
        if tuningProportion > 0:
            numCredibleTune = -1*int(tuningProportion * len(credibleTrainFeatures))
            numMaliciousTune = -1*int(tuningProportion * len(maliciousTrainFeatures))
            
            credibleTuneFeatures = credibleTrainFeatures[numCredibleTune:]
            credibleTuneLabels = credibleTrainLabels[numCredibleTune:]
            credibleTrainFeatures = credibleTrainFeatures[0:numCredibleTune]
            credibleTrainLabels = credibleTrainLabels[0:numCredibleTune]
            
            maliciousTuneFeatures = maliciousTrainFeatures[numMaliciousTune:]
            maliciousTuneLabels = maliciousTrainLabels[numMaliciousTune:]
            maliciousTrainFeatures = maliciousTrainFeatures[0:numMaliciousTune]
            maliciousTrainLabels = maliciousTrainLabels[0:numMaliciousTune]
            
            # merge the credible and malicious back into a validation dataset
            tuneFeatures = np.array(credibleTuneFeatures.tolist() + maliciousTuneFeatures.tolist())
            tuneLabels = np.array(credibleTuneLabels.tolist() + maliciousTuneLabels.tolist())
            
        # merge the malicious and credible back into train/test
        trainFeatures = np.array(credibleTrainFeatures.tolist() + maliciousTrainFeatures.tolist())
        trainLabels = np.array(credibleTrainLabels.tolist() + maliciousTrainLabels.tolist())
        testFeatures = np.array(credibleTestFeatures.tolist() + maliciousTestFeatures.tolist())
        testLabels = np.array(credibleTestLabels.tolist() + maliciousTestLabels.tolist())
        
        return (trainFeatures, trainLabels, testFeatures, testLabels, tuneFeatures, tuneLabels)
    
    
    def getTrainTuneTestWithFold(self, splitValue, tuningProportion=0.0, splitColumn="date"):
        credibleTrainFeatures, credibleTrainLabels, credibleTestFeatures, credibleTestLabels = getClassTrainTestSets(0, splitValue, splitColumn)
        maliciousTrainFeatures, maliciousTrainLabels, maliciousTestFeatures, maliciousTestLabels = getClassTrainTestSets(1, splitValue, splitColumn)
    
        # merge the malicious and credible back into train/test
        trainFeatures = np.array(credibleTrainFeatures.tolist() + maliciousTrainFeatures.tolist())
        trainLabels = np.array(credibleTrainLabels.tolist() + maliciousTrainLabels.tolist())
        testFeatures = np.array(credibleTestFeatures.tolist() + maliciousTestFeatures.tolist())
        testLabels = np.array(credibleTestLabels.tolist() + maliciousTestLabels.tolist())
        
        predefinedSplitTestfold = np.array([])

        if (tuningProportion > 0.0):
            numCredibleTune = int(tuningProportion * len(credibleTrainFeatures))
            numMaliciousTune = int(tuningProportion * len(maliciousTrainFeatures))
            
            credibleClassTestfold = np.array([])
            maliciousClassTestfold = np.array([])
            
            # Top part = earlier articles = training; testfold value = -1
            # Bottom part = later articles = validation; testfold value = 0
            credibleClassTestfold = [-1]*(len(credibleTrainLabels)-numCredibleTune) + [0]*numCredibleTune
            maliciousClassTestfold = [-1]*(len(maliciousTrainLabels)-nummaliciousTune) + [0]*nummaliciousTune
            predefinedSplitTestfold = np.array(credibleClassTestfold + maliciousClassTestfold)

        return (trainFeatures, trainLabels, testFeatures, testLabels, predefinedSplitTestfold)        
        
