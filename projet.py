import csv
import math
import re
from collections import Counter
from datetime import datetime

import xlsxwriter
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def readCSV(filename):
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        codes = []
        docs = []
        for row in csv_reader:
            if row != []:
                if '\xa0' in row[2]:
                    row[2] = row[2].replace(u'\xa0', u' ')
                if 'vide' in row[2]:
                    continue
                if '\n' in row[2]:
                    row[2] = row[2].replace(u'\n', u' ')
                if '\t' in row[2]:
                    row[2] = row[2].replace(u'\t', u'   ')
                temp = row[2]
                row[1] = row[2]
                row.remove(row[2])
                # for r in csv_reader:
                #     if r != []:
                #         if row != r:
                #             if row[0] == r[0]:
                #                 print(r[0])
                #                 temp = temp + " " + r[2]
                # row[2] = temp
                docs.append(row)
                #print(docs)
                line_count += 1
                #print(line_count, row[2])
        d = []
        for row in docs:
            for r in docs:
                if row != r:
                    if row[0] == r[0]:
                        row[1] = row[1] + " " + r[1]
                        #d.append(row)
                        #docs.remove(r)

        # for row in docs:
        #     for r in docs:
        #         if row[0] == r[0]:
        #             row[2] = row[2] + " " + r[2]
        #             docs.remove(r)
        seen = set()
        for row in docs:
            t = row[0]
            if t not in seen:
                seen.add(t)
                d.append(row)
                codes.append(row[0])

        print(f'Processed {line_count} lines.')
        #print(docs)
        #print(len(docs))
    return d, codes


def storeOnlyTexts(doc):
    res = []
    for row in doc:
        #res.append(row[1])
        try:
            if (row != ['\ncote', 'tel', 'tc']) & (row != []):
                res.append(row[1])
        except:
            print(row)
    return res


def TfidVec(doc1, doc2):
    tfidf_vectorizer = TfidfVectorizer()
    documents = []
    documents.append(doc1)
    documents.append(doc2)
    # print(documents)
    result = tfidf_vectorizer.fit_transform(documents)
    return result


def text_to_vector(text):
    words = WORD.findall(str(text))
    return Counter(words)


def cosineSim(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def fillCosineArray(doc):
    cosine = []
    count = 0
    for row in range(len(doc)):
        temp = []
        vec1 = text_to_vector(doc[row][1])
        #print(vec1)
        #print(row)
        #print(len(doc[row]))
        for r in range(len(doc)):
            #print(doc[r][2])
            vec2 = text_to_vector(doc[r][1])
            temp.append(cosineSim(vec1, vec2))

        ind = np.argpartition(temp, -10)[-10:]
        res = []

        for i in ind:
            t = []
            t.append(i)
            t.append(temp[i])
            #print(t)
            res.append(sorted(t))

        #print(res)
        #print(temp)
        cosine.append(res)
        count += 1
        print('row ', count, ' complete')
    return cosine


def writeToExcel(doc, codes, filename, type):
    wb = xlsxwriter.Workbook(filename)
    ws = wb.add_worksheet()
    i = 0
    for row in doc:
        j = 0
        ws.write(i, 0, codes[i])
        j += 1
        print("type = " + str(type))
        for column in row:
            #print(column)
            if type == 0:
                ws.write(i, j, codes[int(column[1])] + ", " + str(column[0]))
            else:
                ws.write(i, j, codes[int(column[0])] + ", " + str(column[1]))
            j += 1
        i += 1
    wb.close()


def doc2vec(doc):
    print('Started Doc2Vec')
    texts = storeOnlyTexts(doc)
    #print(texts[2])
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(texts)]
    max_epochs = 100
    vec_size = 200
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1,
                    workers=16)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")

    result = []

    for i in range(len(doc) - 1):
        similar_doc = model.docvecs.most_similar(i)
        result.append(similar_doc)
    c = 0
    return result


start = datetime.now()
WORD = re.compile(r'\w+')
doc, codes = readCSV('transcriptions-fds-originales.csv')

#cosine = fillCosineArray(doc)
# doc2vecdata = doc2vec(doc)

writeToExcel(fillCosineArray(doc), codes, "cosine.xlsx", 0)
writeToExcel(doc2vec(doc), codes, "doc2vec.xlsx", 1)

end = datetime.now() - start
print(end)