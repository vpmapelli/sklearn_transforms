from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.svm import LinearSVC

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class SimpleImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **args):
        self.si = SimpleImputer(**args)

    def fit(self, X, y=None):
        self.si.fit(X=X)
        return self

    def transform(self, X):
        data = X.copy()
        data = pd.DataFrame.from_records(
                    data = self.si.transform(
                            X=X
                        ),  # o resultado SimpleImputer.transform(<<pandas dataframe>>) é lista de listas
                        columns=X.columns  # as colunas originais devem ser conservadas nessa transformação
                    )
        return data

class NormalizeGrades(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for column in self.columns:
            data.loc[:,column] = X[column].apply(lambda x:x/10)
        return data

class LinearSVCWrapper():
    def __init__(self,**args):
        self.svc_instance = LinearSVC(**args)
    
    def fit(self,*args,**kwargs):
        new_args= list(args)
        new_args[1] = new_args[1].values.ravel()
        return self.svc_instance.fit(*new_args,**kwargs)
    
    def predict(self,X):
        return self.svc_instance.predict(X)
