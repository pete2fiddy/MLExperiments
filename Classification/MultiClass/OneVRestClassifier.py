from Classification.MultiClass.MultiClassifiable import MultiClassifiable
import numpy as np
class OneVRestClassifier(MultiClassifiable):
    '''one vs. rest is apparently susceptible to class imbalance in data'''
    '''All classification models using this must take hyper paramaters
    as a **kwargs and have a "pos_y_val" specifying which value of y
    is the "true" case for that model.
    Also pass the fit_score_func, the method to be called for the particular
    model that will determine the quality of fit between a point and model.
    Must be non-binary so that it can be compared to the output of other models'''
    def __init__(self, X, y, model, fit_score_func, **hyper_params):
        self.X = X
        self.y = y
        self.model = model
        self.fit_score_func = fit_score_func
        self.hyper_params = hyper_params
        self.init_models()

    def init_models(self):
        unique_ys = np.unique(self.y)
        self.models = []
        for i in range(0, unique_ys.shape[0]):
            self.models.append(self.model(self.X, self.y, unique_ys[i], **self.hyper_params))

    def train(self):
        for model in self.models:
            model.train()

    def predict(self, x):
        '''assumes highest fit score is the best to return --
        modify functions where this is not the case so that it is the case
        (easiest is to just take the negative of the fit score)'''
        model_fits = np.zeros((len(self.models)))
        for i in range(0, model_fits.shape[0]):
            model_fits[i] = self.fit_score_func(self.models[i], x)
        best_model_index = np.argmax(model_fits)
        best_model_class = self.models[best_model_index].pos_y_val
        return best_model_class
