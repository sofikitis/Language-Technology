import os
import csv
from nltk.stem import WordNetLemmatizer
import numpy as np
import math

# nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

read_path = 'tag_files/'
entries = os.listdir(read_path)

all_docs = []
for e in entries:
    sub_path = read_path + e

    if os.stat(sub_path).st_size != 0:
        with open(sub_path, encoding="mbcs") as csvFile:
            my_List = list(csv.reader(csvFile, delimiter='\t'))
        csvFile.close()
    else:
        continue

    # Convert list to array
    tagsArr = np.array(my_List)
    all_docs.append(tagsArr)

frequency_matrix = {}
i = 0
for doc in all_docs:
    freq_table = {}
    for tag in doc:
        word = tag[0]
        word = lemmatizer.lemmatize(word)

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

index = {}
for (doc, f_table), ide in zip(tf_idf_matrix.items(), entries):

    for word, weight in f_table.items():
        tup = [str(ide), weight]

        if word in index:
            # append the new doc to the existing array at this slot
            index[word].append(tup)
        else:
            # add the new lemma to the index
            index[word] = [tup]

name = 'index.xml'
with open(name, 'w') as xml_file:
    xml_file.write('<inverted_index>\n')

    for lemma, entries in index.items():
        line = '<lemma name='+'"'+lemma+'"'+'>\n'
        xml_file.write(line)

        for i in entries:
            line = '\t<document id= "' + str(i[0]) + '"' + ' weight= "' + str(i[1]) + '"' + '/>\n'
            xml_file.write(line)

        xml_file.write('</lemma>\n')

    xml_file.write('</inverted_index>')

xml_file.close()
