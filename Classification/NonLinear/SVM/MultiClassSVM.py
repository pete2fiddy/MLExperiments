from Classification.NonLinear.SVM.SupportVectorMachine import SupportVectorMachine
from Classification.MultiClass.OneVRestClassifier import OneVRestClassifier
from Classification.MultiClass.MultiClassifiable import MultiClassifiable
from Classification.Classifiable import Classifiable

class MultiClassSVM(Classifiable):
    '''only uses OneVRest for now.'''
    '''perhaps creating a "multi class" wrapper for
    every ML function is tedious and uneccessary, and instead
    one could init a OneVRest or OneVOne object passing the
    learning model as an argument, and these would be uneccessary

    REASON TO KEEP THIS WAY: Possible for one vs. all for example to have
    all negative outputs, in which case the correct classification would be
    to go for the class closest to the margin (which I suppose would still have
    the same argmax...)'''
    def __init__(self, X, y, **kwargs):
        self.hyper_params = kwargs
        self.X = X
        self.y = y
        self.multi_classifier = OneVRestClassifier(self.X, self.y, SupportVectorMachine, SupportVectorMachine.functional_margin, **self.hyper_params)

    def train(self):
        self.multi_classifier.train()

    def predict(self, x):
        return self.multi_classifier.predict(x)
