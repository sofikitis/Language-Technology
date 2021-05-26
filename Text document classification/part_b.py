import os
import math
import operator
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
regex = re.compile('[^a-zA-Z ]')


def cosine_sim(array_a, array_b):
    sum_ab = 0
    sum_a = 0
    sum_b = 0
    for a, b in zip(array_a, array_b):
        sum_ab = sum_ab + a * b
        sum_a = sum_a + a ** 2
        sum_b = sum_b + b ** 2

    div = (math.sqrt(sum_a) + math.sqrt(sum_b))
    if div != 0:
        sim = sum_ab / (math.sqrt(sum_a) + math.sqrt(sum_b))
    else:
        sim = jaccard_sim(array_a, array_b)

    return sim


def jaccard_sim(array_a, array_b):
    n = 0
    for a, b in zip(array_a, array_b):
        if (a == 0) and (b == 0):
            n = n + 1
        elif (a != 0) and (b != 0):
            n = n + 1
    return n / N


def preprocess_doc(doc_path):
    # open file
    with open(doc_path, 'rb') as cur_file:
        cur_file_to_list = cur_file.read().splitlines()
    # create new doc without the data
    doc_without_data = []
    for i in cur_file_to_list:

        # remove the r' '
        i = str(i)
        i = i[2:]
        i = i[:-1]

        # data lines contain the ':' symbol
        if ':' not in i:
            doc_without_data.append(i)

    doc_without_data = str(doc_without_data)

    # remove punctuation
    doc_without_puncts = regex.sub('', doc_without_data)

    # tokenize the doc
    wordArr = word_tokenize(doc_without_puncts)

    # stem every word
    wordArr = [ps.stem(word) for word in wordArr]

    # remove stop words
    wordArr = [word for word in wordArr if word not in stop_words]
    return wordArr


def calculate_tfidf(all_docs):
    frequency_matrix = {}
    i = 0
    for sentence in all_docs:
        freq_table = {}

        for word in sentence:

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[i] = freq_table
        i = i + 1

    tf_matrix = {}
    i = 0
    for sentence, f_table in frequency_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[i] = tf_table
        i = i + 1

    word_per_doc_table = {}
    for sentence, f_table in frequency_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    total_documents = len(all_docs)

    idf_matrix = {}
    i = 0
    for sent, f_table in frequency_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))

        idf_matrix[i] = idf_table
        i = i + 1

    tf_idf_matrix = {}
    i = 0
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[i] = tf_idf_table
        i = i + 1

    return tf_idf_matrix


def create_vector_space(tfidf_of_a_category_matrix):
    complete_dictionary = {}
    for doc, table in tfidf_of_a_category_matrix.items():
        complete_dictionary.update(table)

    f_vector = {}
    for i in range(1, N):
        cur_max = max(complete_dictionary.items(), key=operator.itemgetter(1))

        f_vector[cur_max[0]] = cur_max[1]
        del complete_dictionary[cur_max[0]]

    return f_vector


def create_vector_for_docs(tfidf_all_docs):
    f_vector_for_all_docs = []

    for doc, table in tfidf_all_docs.items():
        this_docs_vector = []
        for word, value in feature_vector.items():
            if word in table:
                this_docs_vector.append(value)
            else:
                this_docs_vector.append(0)

        f_vector_for_all_docs.append(this_docs_vector)
    return f_vector_for_all_docs


def compute_similarity(cur_vec):
    array_s = []
    for vectorE in vector_space_e:
        sim = cosine_sim(cur_vec, vectorE)
        array_s.append(sim)

    return array_s


def find_most_similar(sim_array):

    position = sim_array.index(max(sim_array))
    return labels_E[position]


N = 5000  # number of features
path = '20_newsgroups/'
entries = os.listdir(path)

all_documents = []
labels = []
label_num = 0

for i in entries:
    label_num = label_num + 1
    sub_path = path + i
    sub_entries = os.listdir(sub_path)

    for j in sub_entries:
        cur_path = sub_path + '/' + j
        cur_doc = preprocess_doc(cur_path)

        all_documents.append(cur_doc)
        labels.append(label_num)

# split document to create set A and set E
set_E, set_A, labels_E, labels_A = train_test_split(all_documents, labels, test_size=0.25)

# calculate tf-idf for all documents in set E and A
tfidf_for_E = calculate_tfidf(set_E)
tfidf_for_A = calculate_tfidf(set_A)

# create N sized feature vector
feature_vector = create_vector_space(tfidf_for_E)

# create vector space for all docs in set E and A
vector_space_e = create_vector_for_docs(tfidf_for_E)
vector_space_a = create_vector_for_docs(tfidf_for_A)

# vector_space_e is N*E matrix (E = # of docs in set E)
# we can find in which category each doc belongs
# with the corresponding value in labels_E.
# vector_space_a is a N*A matrix (A = # of docs in set A)

estimated_labels = []
for vector in vector_space_a:
    similarity_array = compute_similarity(vector)
    est_label = find_most_similar(similarity_array)
    estimated_labels.append(est_label)

# compare estimated_labels with labels_A
print(labels_A)
print(estimated_labels)
print(classification_report(labels_A, estimated_labels, zero_division=0))
