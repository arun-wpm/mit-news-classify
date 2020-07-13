# Evaluation for ImproveTheNews.com news topic classifier
# Analyze the quality of the model and find suitable cutoff for predicted tags

from utilities_v3 import create_one_hot
from predict_topic import probabilities_to_onehot
from sklearn.metrics import roc_auc_score


def auc_roc_eval(actual_topics, y_score):
    num_topics = 594

    y_true = []
    for topics in actual_topics:
        y_true.append(create_one_hot(topics, num_topics))

    return roc_auc_score(y_true, y_score)


def false_eval(actual_topics, probabilities):
    predicted_topics = probabilities_to_onehot(probabilities)
    total_actual_topics = 0
    false_positives = 0
    true_positives = 0

    for i in range(len(actual_topics)):
        total_actual_topics += len(actual_topics[i])
        for topic in predicted_topics[i]:
            if str(topic) in actual_topics[i]:
                true_positives += 1
            else:
                false_positives += 1

    false_negatives = total_actual_topics - true_positives
    return total_actual_topics, false_negatives, false_positives
