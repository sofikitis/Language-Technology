import xml.etree.ElementTree as ET
import time
import random
from operator import itemgetter
import numpy as np

# φορτωση xml σε dict
inverted_index = r"index.xml"

tree = ET.parse(inverted_index)
all_lemmas = tree.findall("./lemma")

index_dictionary = {}
words_for_eval = []

# iterate trough all the lemma tags
i = 0
for cur_lemma in all_lemmas:
    name = cur_lemma.get('name')
    cur_documents = cur_lemma.findall("./document")

    # select random 150 words for the queries
    r = random.randint(1, 100)
    if i <= 150 and r < 10:
        words_for_eval.append(name)
        i += 1

    # iterate trough all the document tags for every lemma
    for doc in cur_documents:
        ide = doc.get('id')
        w = doc.get('weight')
        tup = [int(ide), float(w)]

        if name in index_dictionary:
            index_dictionary[name].append(tup)
        else:
            index_dictionary[name] = [tup]

# Αξιολόγηση ευρετηρίου.
Q = 100

# start timer
start = time.time()

# queries with 1 word
for i in range(1, 20):
    r = random.randint(1, 150)
    query = words_for_eval[r]

    results = index_dictionary[query]
    results = np.array(sorted(results, key=itemgetter(1), reverse=True))
    print(results[:, 0])

# queries with 2 words
for i in range(1, 20):
    r = random.randint(1, 150)
    query = words_for_eval[r]
    results1 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results2 = np.array(index_dictionary[query])

    # find the common ids in the results of the two queries
    common_ids = np.intersect1d(results1[:, 0], results2[:, 0], assume_unique=True)

    results = []
    for ide in common_ids:
        # find the position of the common ids in every results array
        pos1 = np.where(results1[:, 0] == ide)[0][0]
        pos2 = np.where(results2[:, 0] == ide)[0][0]

        w = results1[pos1, 1] + results2[pos2, 1]
        tup = [ide, w]
        results.append(tup)

    if not results:
        print("No documents with those queries")
    else:
        results = np.array(sorted(results, key=itemgetter(1), reverse=True))
        print(results[:, 0])

# queries with 3 words
for i in range(1, 30):
    r = random.randint(1, 150)
    query = words_for_eval[r]
    results1 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results2 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results3 = np.array(index_dictionary[query])

    # find the common ids in the results of the three queries
    intersection = np.intersect1d(results1[:, 0], results2[:, 0], assume_unique=True)
    common_ids = np.intersect1d(intersection, results3[:, 0], assume_unique=True)

    results = []
    for ide in common_ids:
        # find the position of the common ids in every results array
        pos1 = np.where(results1[:, 0] == ide)[0][0]
        pos2 = np.where(results2[:, 0] == ide)[0][0]
        pos3 = np.where(results3[:, 0] == ide)[0][0]

        w = results1[pos1, 1] + results2[pos2, 1] + results3[pos3, 1]
        tup = [ide, w]
        results.append(tup)

    if not results:
        print("No documents with those queries")
    else:
        results = np.array(sorted(results, key=itemgetter(1), reverse=True))
        print(results[:, 0])

# queries with 4 words
for i in range(1, 30):
    r = random.randint(1, 150)
    query = words_for_eval[r]
    results1 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results2 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results3 = np.array(index_dictionary[query])

    r = random.randint(1, 150)
    query = words_for_eval[r]
    results4 = np.array(index_dictionary[query])

    # find the common ids in the results of the four queries
    intersection1 = np.intersect1d(results1[:, 0], results2[:, 0], assume_unique=True)
    intersection2 = np.intersect1d(results3[:, 0], results4[:, 0], assume_unique=True)
    common_ids = np.intersect1d(intersection1, intersection2, assume_unique=True)

    results = []
    for ide in common_ids:
        # find the position of the common ids in every results array
        pos1 = np.where(results1[:, 0] == ide)[0][0]
        pos2 = np.where(results2[:, 0] == ide)[0][0]
        pos3 = np.where(results3[:, 0] == ide)[0][0]
        pos4 = np.where(results4[:, 0] == ide)[0][0]

        w = results1[pos1, 1] + results2[pos2, 1] + results3[pos3, 1] + results4[pos4, 1]
        tup = [ide, w]
        results.append(tup)

    if not results:
        print("No documents with those queries")
    else:
        results = np.array(sorted(results, key=itemgetter(1), reverse=True))
        print(results[:, 0])

# end timer
end = time.time()
tt = end - start

print("Average response time: ", tt/Q)
