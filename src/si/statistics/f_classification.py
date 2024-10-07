<<<<<<< HEAD
from si.data.dataset import Dataset  
import scipy.stats  # Importação corrigida

def f_classification(dataset: Dataset) -> tuple:
    classes = dataset.get_classes()
    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[mask, :]
        groups.append(group)
    
    F, p = scipy.stats.f_oneway(*groups)
    return F, p  # Retorna F e p separadamente
=======


from si.data.dataset import Dataset
import scipy


def f_classification(dataset: Dataset) -> tuple:
    
    classes = dataset.get_classes()

    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[ mask , :]
        groups.append(group)

    return scipy.stats.f_oneway(*groups)
>>>>>>> 1277ea0d469379272f7a89a41c1f53b396d113e1
