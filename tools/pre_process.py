from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix

def build_data():
    rows = []
    with open('input_data') as fin:
        for line in fin:
        	data = line.split(':')[0]
        	label = line.split(':')[1].split('\n')[0]
        	rows.append((data, label))
    return rows

def write_to_file(filename, matrix, label_list):
	with open(filename, 'w') as fout:
			k = 0
			for command in matrix:
				fout.write(str(label_list[k]) + ' ')
				index = 1
				for term in command:
					if term != 0.0:
						fout.write(str(index) + ':' + str(term) + ' ')
					index = index + 1
				fout.write('\n')
				k = k + 1

data = build_data()
text_list = []
label_list = []
for (text, label) in data:
	text_list.append(text)
	label_list.append(label)

vectorizer = CountVectorizer()
vectorizer.fit_transform(text_list)
transformer = TfidfVectorizer()
tfidf_matrix =  transformer.fit_transform(text_list).toarray()

testSplit = int(.8 * len(tfidf_matrix))
training_set = tfidf_matrix[:testSplit]
training_labels = label_list[:testSplit]

testing_set = tfidf_matrix[testSplit:]
testing_labels = label_list[testSplit:]

write_to_file('training_data_vector', training_set, training_labels)
write_to_file('testing_data_vector', testing_set, testing_labels)
