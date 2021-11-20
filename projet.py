import numpy as np
import pandas as pd
import utils
from sklearn.metrics import recall_score, precision_score,confusion_matrix


#Question 1 
def getPrior(train):
  """ Fonction qui renvoie un dictoinnaire contenant
  l'estimation de target ainsi que les valeures superieures
  et inferieures de l'interval de confiance"""

  dic={} #Initialisation du dictionnaui

  # Formule d'interval de confiance:
  # ci = moyenne(target) +/- z* (ecart-type/ sqrt(n))

  #Determination de z 
  # pour un interval de confiance de 0.95 
  z= 1.96 

  dic["estimation"]= train.target.mean() #Moyenne 
  dic["min5pourcent"] = train.target.mean() - z*(train.target.std()/np.sqrt(train.shape[0])) #born inf
  dic["max5pourcent"] = train.target.mean() + z*(train.target.std()/np.sqrt(train.shape[0])) #borne sup

  return dic


#Question 2

class APrioriClassifier(utils.AbstractClassifier):
  """ classe qui hérite de la classe AbstractClassifier, elle estime la classe de chaque individu par la classe majoritaire"""
  def __init__(self):
    pass

  #Question 2a
  def estimClass(self, dic):
    """ Fonction qui estime la classe majoritaire à partir d'un dictionnaire passé en argument"""
    #Cette fonction revoie la même classe peut importe l'argument dic, qui 
    #vaut 1 pour ce classifieur a priori 
    return 1

  #Question 2b
  def statsOnDF(self, dataf):
    """Fonction qui determine les fauc négatives et positivies, 
    vrai positives et négatives ainsi que le rappel et la precision"""
    dico= {"VP": 0,"VN":0, "FP":0, "FN":0 } #Initialisation des valeures à 0
    for i in range(dataf.shape[0]): #Parcourir tout les n-uplets
      dic= utils.getNthDict(dataf, i) #Recuperer le iéme n-uplet 
      predict= self.estimClass(dic)
      #Traitements de differents cas de VP, VN, FP, FN
      if dic["target"]==1 and predict ==1:
        dico["VP"]+=1 
      elif dic["target"]==0 and predict==0: 
        dico["VN"]+=1
      elif dic["target"]==0 and predict==1: 
        dico["FP"]+=1
      else: 
        dico["FN"]+=1
      
      #Rappel = vp/(vp+fn)
      rappel= dico["VP"] / (dico["VP"] +dico["FN"] )

      #Precison = vp/(vp+fp)
      precision= dico["VP"] / (dico["VP"] +dico["FP"] )

    return {"VP": dico["VP"], "VN":dico["VN"], "FP": dico["FP"], "FN": dico["FN"], "rappel": rappel,  "précision": precision}

  #Autre methode en utilisant la bibliotheque scikit-learn 
  # def statsOnDF(self, dataf): 
  #   pred= np.repeat(1, dataf.shape[0], axis=0)
  #   cf= confusion_matrix(pred, dataf.target) #Matrice preresentant les vp, vn, fp, fn
  #   dic={} 

  #   #Affichage des valeures dans la matrice 
  #   dic["VN"]= cf[0][0]
  #   dic["FN"]=cf[0][1]

  #   dic["FP"]=cf[1][0]
  #   dic["VP"]=cf[1][1]
  #   dic["precision"]= precision_score(dataf.target, pred)
  #   dic["rappel"]= recall_score( dataf.target, pred)
  #   return dic
 
#Question 3a 
def P2D_l(df,attr): 
  """Fonction qui determine la probabilité conditionelle pour les valeures 
  d'un rribut passé en argument sachant les differentes valeures de target possible dans le dataframe"""
  dico={} #Initialisation du dictionnaire résultat (renvoyé par la fonction)
  crt=pd.crosstab(df.target,df[attr], margins=True ) #Creation de la matrice crosstable qui aura les occurences de chaque rribut répartis par atasse (1 ou 0)
  for tgt in df.target.unique(): #Parcourir les targets 
    dic={} #Initialisation du dictionnaire des probabilités de chaque valeur de l'rribut sachant le target tgt
    for at in df[attr].unique(): #Parcourir les éléments de l'rribut 
      dic[at]=crt[at][tgt]/crt["All"][tgt] #Calculer la probabilité conditionelle   
    dico[tgt]=dic #Aout du dictionnaire de probabilité au dictionnaire résultat 
  return dico  