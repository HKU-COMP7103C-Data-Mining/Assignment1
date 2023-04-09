import os, torch
from keras_preprocessing.sequence import pad_sequences


def load_vocab():
    file_path = os.path.join('aclImdb', 'imdb.vocab')
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    vocab = set()
    for line in lines:
        word = line.strip('\n')
        vocab.add(word)
    return vocab


def load_data(flag, vocab_set):
    file_path = os.path.join('aclImdb', flag)
    res = []
    labels_dict = {'neg': 0, 'pos': 1}
    sentences = []
    labels = []
    for label in labels_dict.keys():
        for file_name in os.listdir(os.path.join(file_path, label)):
            with open(os.path.join(os.path.join(file_path, label), file_name), encoding='utf-8') as f:
                sentence = f.readline()
                sentence = sentence.lower()
                words_temp = sentence.split(' ')
                words = []
                for word in words_temp:
                    if word in vocab_set:
                        words.append(word)
                    else:
                        words.append('<UNK>')
                sentences.append(words)
                labels.append(labels_dict[label])
    return sentences, labels


# batch generator
def generate_batch(df, size=32):
    tokens_list = []
    label_list = []
    for item in df:
        tokens_list.append(item[0])
        label_list.append(item[1])

        if len(tokens_list) == size:
            yield tokens_list, label_list
            tokens_list = []
            label_list = []
    if len(tokens_list) != 0:
        yield tokens_list, label_list


# padding
# !pip install keras_preprocessing
def prepare_sequences(sequences, index):
    idxs = [[index[w] for w in seq] for seq in sequences]
    max_len = max([len(seq) for seq in sequences])
    # if len(index) == len(words_index):
    idxs = pad_sequences(maxlen=max_len, sequences=idxs, padding="post", value=index['<PAD>'])
    # else:
    #     idxs = pad_sequences(maxlen=max_len, sequences=idxs, padding="post", value=-1)
    return torch.tensor(idxs, dtype=torch.long)


# def rank_sentences(sentences):
#     ranked_sentences = sorted(sentences, key=len, reverse=True)
#     ranked_lengths = [len(sentence) for sentence in ranked_sentences]
#     return ranked_sentences, ranked_lengths

def predict(model, df_test, device, words_index):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (sentences, label) in enumerate(generate_batch(df_test, 1)):

            sentence_in = prepare_sequences(sentences, words_index).to(device)
            # targets = torch.tensor(label, dtype=torch.float).to(device)

            y_pred = model(sentence_in)
            for y in y_pred:
                if y.cpu().numpy()[0] > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
    return predictions


def compute_metrics(predictions, test_labels):
    tp = 0  # True Positive
    tn = 0  # True Negative
    fp = 0  # False Positive
    fn = 0  # False Negative
    for i in range(len(predictions)):
        if predictions[i] == 1 and test_labels[i] == 1:
            tp = tp + 1
        elif predictions[i] == 0 and test_labels[i] == 0:
            tn = tn + 1
        elif predictions[i] == 1 and test_labels[i] == 0:
            fp = fp + 1
        else:
            fn = fn + 1

    acc = (tp + tn) / len(test_labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, precision, recall, f1
