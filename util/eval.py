from sklearn.metrics import classification_report, confusion_matrix


def get_classification_report(y_true, y_prediction):
  cr = classification_report(y_true, y_prediction)
  return cr


def get_confusion_matrix(y_true, y_prediction):
  cm = confusion_matrix(y_true, y_prediction)
  return cm