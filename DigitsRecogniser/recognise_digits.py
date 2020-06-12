import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model

models = ["CNN", "XGBOOST", "Softmax", "RandomForest", "GNB"]
files = ["CNN.h5", "XGBOOST.bin", "Softmax.pkl", "RandomForest.pkl", "GNB.pkl"]

# Digits array contains digits of grid in column major order
def predictDigits(digits: np.ndarray, model: int):
    assert (len(digits) == 81) and (model>=0) and (model<=4)

    res = ""
    for i in range(9):
        for j in range(9):
            digit = predictDigit(digits[j*9 + i], model)
            res += str(digit)

    return res
    #return "009000780830019000610000403001900027000040000590008300905000072000590048082000900"


def preprocess(x_vec):
    x_vec = x_vec.astype(np.uint8)
    x_vec = 255 - x_vec
    hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (14, 14), 12)
    return hog.compute(x_vec).reshape(1, -1)

# Take decision based on probability vector of each class
# cost_r: Cost of rejection (In our case we will mark it as empty or no digit = 0 in Sudoku)
# cost_w: Cost of wrong classification (In our case we will mark it as empty or no digit = 0 in Sudoku)
def take_decision(probabilities, cost_r=10, cost_w=30):
  assert cost_w != 0

  # Reference: https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/dtheory.pdf
  pred_class = np.argmax(probabilities)
  if(probabilities[pred_class] > (1 - (cost_r/cost_w))):
    return pred_class

  # reject => No digit => 0 for our case
  return 0


def preprocess_for_CNN(digit: np.ndarray):
  digit = 255 - digit
  digit = digit/255
  digit = digit.reshape((1, 28, 28, 1))
  return digit

def predictDigit(digit: np.ndarray, model):
    # Less than 10 pixels coloured
    if np.sum(digit) < 10*255:
        return 0
    
    if models[model]=="XGBOOST":
      clf = xgb.XGBClassifier(objective="multi:softmax", booster="gbtree", num_classes=10, )
      clf.load_model("DigitsRecogniser/models/" + files[model])
      prob = clf.predict_proba(preprocess(digit))
      prob = np.array([x[1] for x in prob])
    elif models[model]=="CNN":
      clf = load_model("DigitsRecogniser/models/" + files[model])
      processed_digit = preprocess_for_CNN(digit)
      prob = clf.predict(processed_digit)
      prob = prob[0]
    else:
      clf = joblib.load("DigitsRecogniser/models/" + files[model])
      prob = clf.predict_proba(preprocess(digit))
      prob = prob[0]
    
    return take_decision(prob, 8, 10)
