#!/usr/bin/env python
# coding: utf-8

# In[1]:


data_dir = r"C:\Users\Admin\Desktop\Untitled Folder 1\Machine-Learning-for-Security-Analysts\Malicious URLs.csv"


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("Libraries Imported")


# In[3]:


print("Loading CSV Data")
url_df = pd.read_csv(data_dir)
test_url = url_df['URLs'][4]
print("CSV Data Loaded")


# In[4]:


print(url_df)


# In[5]:


test_percentage = .2

train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)

labels = train_df['Class']
test_labels = test_df['Class']
print("Split Complete")


# In[6]:


print("- Counting Splits -")
print("Training Samples:", len(train_df))
print("Testing Samples:", len(test_df))

count_train_classes = pd.value_counts(train_df['Class'])
count_train_classes.plot(kind='bar', fontsize=16)
plt.title("Class Count (Training)", fontsize=20)
plt.xticks(rotation='horizontal')
plt.xlabel("Class", fontsize=20)
plt.ylabel("Class Count", fontsize=20)
plt.show()

count_test_classes = pd.value_counts(test_df['Class'])
count_test_classes.plot(kind='bar', fontsize=16, colormap='ocean')
plt.title("Class Count (Testing)", fontsize=20)
plt.xticks(rotation='horizontal')
plt.xlabel("Class", fontsize=20)
plt.ylabel("Class Count", fontsize=20)

plt.show()


# In[7]:


def tokenizer(url):
    tokens = re.split('[/-]', url) 
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            if "com" in dot_split:                
                dot_split.remove("com")                
            if "www" in dot_split:
                dot_split.remove("www")
            tokens += dot_split
    
    return tokens
    
print("Tokenizer defined")


# In[8]:


print("Test URL")
print(test_url)

print("Tokenized Output")
tokenized_url = tokenizer(test_url)
print(tokenized_url)


# In[9]:


print("- Training Count Vectorizer -")
cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs'])

print("Training TF-IDF Vectorizer")
tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(train_df['URLs'])

print("Vectorizing Complete")


# In[10]:


print("- Count Vectorizer -")
test_count_X = cVec.transform(test_df['URLs'])

print("TFIDF Vectorizer")
test_tfidf_X = tVec.transform(test_df['URLs'])

print("Vectorizing Complete")


# In[11]:


def generate_report(cmatrix, score, creport):
    plt.figure(figsize=(5,5))
    sns.heatmap(cmatrix, 
              annot=True, 
              fmt="d", 
              linewidths=.5, 
              square = True,
              cmap = 'Blues', 
              annot_kws={"size": 16},    
              xticklabels=['bad', 'good'],
              yticklabels=['bad', 'good']
              )
    plt.xticks(rotation='horizontal', fontsize=16)
    plt.yticks(rotation='horizontal', fontsize=16)
    plt.xlabel('Actual Label', size=20);
    plt.ylabel('Predicted Label', size=20);
    title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(title, size = 20);
    print(creport)
    plt.show()
    
print("Report Generator Defined ")


# In[12]:


mnb_tfidf = MultinomialNB(alpha=0.1)
mnb_tfidf.fit(tfidf_X, labels)
score_mnb_tfidf = mnb_tfidf.score(test_tfidf_X, test_labels)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfidf_X)
       
cmatrix_mnb_tfidf = confusion_matrix(predictions_mnb_tfidf, test_labels)
creport_mnb_tfidf = classification_report(predictions_mnb_tfidf, test_labels)

print("Model Built")
generate_report(cmatrix_mnb_tfidf, score_mnb_tfidf, creport_mnb_tfidf)


# In[13]:


mnb_count = MultinomialNB()
mnb_count.fit(count_X, labels)

score_mnb_count = mnb_count.score(test_count_X, test_labels)
predictions_mnb_count = mnb_count.predict(test_count_X)
cmatrix_mnb_count = confusion_matrix(predictions_mnb_count, test_labels)
creport_mnb_count = classification_report(predictions_mnb_count, test_labels)

print("Model Built ")
generate_report(cmatrix_mnb_count, score_mnb_count, creport_mnb_count)


# In[14]:


lgs_tfidf = LogisticRegression(solver='lbfgs',max_iter =100)
lgs_tfidf.fit(tfidf_X, labels)

score_lgs_tfidf = lgs_tfidf.score(test_tfidf_X, test_labels)
predictions_lgs_tfidf = lgs_tfidf.predict(test_tfidf_X)
cmatrix_lgs_tfidf = confusion_matrix(predictions_lgs_tfidf, test_labels)
creport_lgs_tfidf = classification_report(predictions_lgs_tfidf, test_labels)

print("Model Built")
generate_report(cmatrix_lgs_tfidf, score_lgs_tfidf, creport_lgs_tfidf)


# In[15]:


lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)

score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)
cmatrix_lgs_count = confusion_matrix(predictions_lgs_count, test_labels)
creport_lgs_count = classification_report(predictions_lgs_count, test_labels)

print("Model Built")
generate_report(cmatrix_lgs_count, score_lgs_count, creport_lgs_count)


# In[13]:


with open('model.pkl','wb') as file:
    pickle.dump(mnb_tfidf,file)


# In[14]:


arr=["web.me.com/ccling/My_research/publication.html",
    "https://8996dcbb4791.ngrok.io/",
    "mphtadhci5mrdlju.tor2web.org/"]
link="web.me.com/ccling/My_research/publication.html"
arr.insert(len(arr),link)
print(arr)
arr = tVec.transform(arr)
score = mnb_tfidf.predict(arr)
print(score)


# In[14]:


with open('model.pkl','rb') as file:
    data = pickle.load(file)


# In[15]:


import flask


# In[19]:


app = flask.Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET','POST'])

def main():
    arr=[]
    if flask.request.method =='GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method =='POST':       
        link=flask.request.form['link']
        arr.append(link)
        print(arr)
        arr = tVec.transform(arr)
        score = data.predict(arr)
        print(score)
        return flask.render_template('main.html',output='The given link is {}'.format(score))
    
    
    
if __name__ == '__main__':
    app.run()








