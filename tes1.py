from operator import truth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz 
import re
import nltk
# nltk.download('punkt')
import ast
import string
import itertools

from datetime import datetime,timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer


'''=================== Import Data ===================='''
df = pd.read_csv('data.csv')

jakarta = pytz.timezone('Asia/Jakarta')
#df['datetime_created'] = df['created_at'].apply(lambda x: datetime.strptime(x,'%a %b %d %H:%M:%S %z %Y').replace(tzinfo=pytz.UTC).astimezone(jakarta))
#df['date_created'] = df['datetime_created'].apply(lambda x: x.date())
#df['time_created'] = df['datetime_created'].apply(lambda x: x.time())
#df = df.drop(['datetime_created'],axis=1)
df.head(900)

'''=================== Pre-Processing ( Cleanning , Stemming , Stopword Removal , TF-IDF ) ===================='''
# Removing Duplicate if any
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Clean text
len(df[df['clean_text'].isnull()==True])
pd.set_option('display.max_colwidth', None)
df[df['clean_text'].isnull()==True]['original_text']

# Removing Observation
df = df.dropna(subset=['clean_text'])
df = df.reset_index(drop=True)

df.isnull().sum()

df[df['lang']!='in']['lang'].value_counts()

# some word not in Indonesian
def trans(x,src):
    translator = Translator()
    try:
        sentence = translator.translate(x, src=src,dest='id').text
    except:
        sentence = x
    return sentence

def repair_exaggeration(x):
    word_tokens = word_tokenize(x)
    new_x =''
    for i in word_tokens:
        if (i =='ekspedisi'):
            new = re.sub(r'(\w)\1\1+',r'\1\1',i)
            new_x = new_x +new+' '
        elif(i =='ekspress'):
            new = 'kurir'
            new_x = new_x +new+' '
        else:
            new = re.sub(r'(\w)\1\1\1+',r'\1',i)
            new_x = new_x +new+' '
    return new_x

def del_word(x,key_list):
    n = len(key_list)
    word_tokens = word_tokenize(x)
    new_x =''
    for word in word_tokens:
        if word not in key_list:
            new_x = new_x+word+' '
    return new_x

def clean_tweets(tweet):
    # nltk.download('stopwords')
    my_file = open("./dataset/cleaning_source/combined_stop_words.txt", "r")
    content = my_file.read()
    stop_words = content.split("\n")
    file_2  = open("./dataset/cleaning_source/update_combined_slang_words.txt", "r")
    content2 = file_2.read()
    slang_words = ast.literal_eval(content2)
    my_file.close()
    file_2.close()

    tweet = tweet.lower()
    #after tweepy preprocessing the colon left remain after removing mentions
    #or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    #remove emojis from tweet
    #tweet = emoji_pattern.sub(r'', tweet)
    
    #remove punctuation manually
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    
    #remove tags
    tweet=re.sub("&lt;/?.*?&gt;","&lt;&gt;",tweet)
    
    #remove digits and special chars
    tweet=re.sub("(\\d|\\W)+"," ",tweet)

    #remove other symbol from tweet
    tweet = re.sub(r'â', '', tweet)
    tweet = re.sub(r'€', '', tweet)
    tweet = re.sub(r'¦', '', tweet)

    word_tokens = word_tokenize(tweet)
    for w in word_tokens:
        if w in slang_words.keys():
            word_tokens[word_tokens.index(w)] = slang_words[w]

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in string.punctuation:
            filtered_tweet.append(w.lower())
    return ' '.join(filtered_tweet)

def count_words(x):
    words = word_tokenize(x)
    n=len(words)
    return n

df['clean_text'] = df.apply(lambda x: trans(x['clean_text'],x['lang']) if(x['lang']!='in') else x['clean_text'],axis=1)
clean_text = df['clean_text'].copy()
pd.set_option('display.max_colwidth', 100)
clean_text.tail(15)

# Cleaning text that exaggerate the typing such as 'ekspedisi' , 'ekspress', 'JNE', 'Sicepat', 'JNT'
clean_text_exag = clean_text.apply(lambda x: repair_exaggeration(x))
clean_text_exag.tail(15)

# Recleaning after remove exaggeration
re_clean = clean_text_exag.apply(lambda x: clean_tweets(x))

# keywords for querying data, need to remove them
keyword = ['jne','jnt','sicepat','ekspedisi','indonesia','sigesit','jnecare','ekspress']
clean_text_extra = re_clean.apply(lambda x: del_word(x,keyword))
clean_text_extra.tail(15)
clean_text_extra.to_csv("./dataset/clean_text_extra.tail.csv")

df['clean_text'] = clean_text_extra
df['word_length'] = df['clean_text'].apply(lambda x:count_words(x))
df['word_length'].value_counts().sort_index()
df = df.drop(df[df['word_length']==0].index,axis=0)
df = df.reset_index(drop=True)

'''=================== Word Processing ( Stemming , Stopword ) ===================='''
# Create word dictionary
word_dict = {}
for i in range(0,len(df['clean_text'])):
    sentence = df['clean_text'][i]
    word_token = word_tokenize(sentence)
    for j in word_token:
        if j not in word_dict:
            word_dict[j] = 1
        else:
            word_dict[j] += 1

len(word_dict)
len({k:v for (k,v) in word_dict.items() if v < 4})

# Import dataset and read by Lexicon data to remove negation words
negasi = ['bukan','tidak','ga','gk']
lexicon = pd.read_csv('./dataset/lexicon/modified_full_lexicon.csv')
lexicon = lexicon.drop(lexicon[(lexicon['word'] == 'bukan')
                               |(lexicon['word'] == 'tidak')
                               |(lexicon['word'] == 'ga')|(lexicon['word'] == 'gk') ].index,axis=0)
lexicon = lexicon.reset_index(drop=True)
len(lexicon)
lexicon.head(10)
lexicon_word = lexicon['word'].to_list()
lexicon_num_words = lexicon['number_of_words']
len(lexicon_word)

# Checking if there is words in dictionary that does not included in the lexicon
ns_words = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
for word in word_dict.keys():
    if word not in lexicon_word:
        kata_dasar = stemmer.stem(word)
        if kata_dasar not in lexicon_word:
            ns_words.append(word)
len(ns_words)

len({k:v for (k,v) in word_dict.items() if ((k in ns_words)&(v>3)) })
ns_words_list = {k:v for (k,v) in word_dict.items() if ((k in ns_words)&(v>3))}

# # It turns out that the words that is not included in lexicon
# sort_orders = sorted(ns_words_list.items(), key=lambda x: x[1], reverse=True)
# sort_orders=sort_orders[0:20]
# for i in sort_orders:
#     print(i[0], i[1])

# word_to_plot = df['clean_text'].copy()
# word_to_plot_1 = word_to_plot.apply(lambda x: del_word(x,negasi))

# # """creating word cloud to see what kind of words that appear often in the tweets related to the Dataset Couirier JNE, JNT and Sicepat"""

# # wordcloud = WordCloud(width = 800, height = 800, background_color = 'black', max_words = 1000
# #                       , min_font_size = 20).generate(str(word_to_plot_1))
# # #plot the word cloud
# # fig = plt.figure(figsize = (8,8), facecolor = None)
# # plt.imshow(wordcloud)
# # plt.axis('off')
# # plt.show()

'''=================== Clustering K-Means ===================='''

lexicon['number_of_words'].value_counts()
'pekerti' in word_dict # return False
'budi baik' in lexicon_word # Return True

# calculating the sentiment of words by mathing them to the lexicon while also creating the bag of words matrix
sencol =[]
senrow =np.array([])
nsen = 0
factory = StemmerFactory()
stemmer = factory.create_stemmer()
sentiment_list = []

# function to write the word's sentiment if it is founded
def found_word(ind,words,word,sen,sencol,sentiment,add):
    # if it is already included in the bag of words matrix, then just increase the value
    if word in sencol:
        sen[sencol.index(word)] += 1
    else:
        #if not, than add new word
        sencol.append(word)
        sen.append(1)
        add += 1
    #if there is a negation word before it, the sentiment would be the negation of it's sentiment
    if (words[ind-1] in negasi):
        sentiment += -lexicon['weight'][lexicon_word.index(word)]
    else:
        sentiment += lexicon['weight'][lexicon_word.index(word)]
    
    return sen,sencol,sentiment,add

# checking every words, if they are appear in the lexicon, and then calculate their sentiment if they do
for i in range(len(df)):
    nsen = senrow.shape[0]
    words = word_tokenize(df['clean_text'][i])
    sentiment = 0 
    add = 0
    prev = [0 for ii in range(len(words))]
    n_words = len(words)
    if len(sencol)>0:
        sen =[0 for j in range(len(sencol))]
    else:
        sen =[]
    
    for word in words:
        ind = words.index(word)
        # check whether they are included in the lexicon
        if word in lexicon_word :
            sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
        else:
        # if not, then check the root word
            kata_dasar = stemmer.stem(word)
            if kata_dasar in lexicon_word:
                sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
        # if still negative, try to match the combination of words with the adjacent words
            elif(n_words>1):
                if ind-1>-1:
                    back_1    = words[ind-1]+' '+word
                    if (back_1 in lexicon_word):
                        sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
                    elif(ind-2>-1):
                        back_2    = words[ind-2]+' '+back_1
                        if back_2 in lexicon_word:
                            sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
    # if there is new word founded, then expand the matrix
    if add>0:  
        if i>0:
            if (nsen==0):
                senrow = np.zeros([i,add],dtype=int)
            elif(i!=nsen):
                padding_h = np.zeros([nsen,add],dtype=int)
                senrow = np.hstack((senrow,padding_h))
                padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
                senrow = np.vstack((senrow,padding_v))
            else:
                padding =np.zeros([nsen,add],dtype=int)
                senrow = np.hstack((senrow,padding))
            senrow = np.vstack((senrow,sen))
        if i==0:
            senrow = np.array(sen).reshape(1,len(sen))
    # if there isn't then just update the old matrix
    elif(nsen>0):
        senrow = np.vstack((senrow,sen))
        
    sentiment_list.append(sentiment)

len(sentiment_list)
# print(senrow.shape[0])

# data frame that contain bag of words and the sentiments that have been calculated before"""
sencol.append('sentiment')
sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
sentiment_data = np.hstack((senrow,sentiment_array))
df_sen = pd.DataFrame(sentiment_data,columns = sencol)
# print(df_sen.head(100))

# show if the sentiment is correct by looking at the original text
cek_df = pd.DataFrame([])
cek_df['text'] = df['original_text'].copy()
cek_df['sentiment']  = df_sen['sentiment'].copy()
print(cek_df.head(900))
# cek_df.to_csv("result.csv")

'''=================== Classification Using SVM ===================='''
df['sentiment'] = df_sen['sentiment']
# df.head(900)
# print(cek_df)

def conf_mat(data, label) : 
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    svclassifier = svm.SVC(kernel='linear') # Linear Kernel
    # svclassifier = svm.SVC(kernel='rbf') # Gaussian Kernel
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)

    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]


count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(cek_df['text'])
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

# data = 
label = np.where(cek_df['sentiment'] > 0, 1, 0)
# print(data)
# print(label)
conf_mat(indexed_data,label)

'''=================== Result Graph ===================='''
# show graph
def GraphSentimentDistribution(data,title):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.kdeplot(data,color='m',shade=True)
    plt.title(title)
    plt.xlabel('sentiment')

def GraphAxes(data):
    sns.set(style="whitegrid") 
    sns.boxplot(x=data)

def GraphDataFeature(data):
    plt.figure(figsize=(15,15))
    h = sns.heatmap(data, annot=True,vmin=-1, vmax=1, center= 0)
    plt.show()

def GraphTop15MostOftenOccuredWords(dataX, dataY, title):
    #pal =sns.dark_palette("purple", input="xkcd",n_colors=15)
    pal =sns.light_palette("navy", reverse=True,n_colors=15)
    g = sns.barplot(x = dataX, y = dataY, palette=pal)
    g.grid(False)
    plt.xlabel('Occurences')
    plt.ylabel('Words')
    plt.title(title, fontweight='bold') 
    for i in range(15):
        g.text(dataX[i],i+0.22, dataX[i],color='black')
    plt.show()

def GraphNumericalFeature(df_corr, matrix):
    plt.figure()
    plt.title('correlation between numerical data',fontweight='bold')
    # df_corr = df.corr()
    # matrix = np.triu(df.corr())
    cmap =  sns.cubehelix_palette(light=0.5, as_cmap=True)
    h = sns.heatmap(df_corr, annot=True, vmin=-1, vmax=1, center= 0, mask=matrix, cmap = cmap)
    plt.show()

def GraphSentimentEveryLanguage(dataDF,dataX,dataY,title):
    sns.set(style="white", color_codes=True)
    plt.figure(figsize=(10,8))
    plt.title(title, fontweight='bold')
    l = sns.boxplot(x=dataX, y=dataY, data=dataDF, palette= sns.color_palette("RdPu", 10))

def GraphPossiblySensitive(dataDF,dataX,dataY):
    plt.figure(figsize=(8,8))
    g = sns.boxplot(data=dataDF, x=dataX, y=dataY)
    plt.show()

def GraphMostPositiveNegativeSentimentPlace(dataPos,dataNeg,dataX,dataY):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,10))
    fig.suptitle('Most Positive and Most Negative Sentiment Place',fontweight='bold')
    h = sns.barplot(x=dataX, y=dataY, data=dataPos, ax=ax1, palette=sns.color_palette("Blues_d",n_colors=10))
    n = sns.barplot(x=dataX, y=dataY, data=dataNeg, ax=ax2, palette=sns.color_palette('RdPu_d',n_colors=10))
    ax1.set_title('Top 10 Positive')
    ax2.set_title('Top 10 Negative')
    plt.show()

def GraphTop10TweetPlace(data,dataX,dataY, title):
    pal =sns.dark_palette("green", input="xkcd",n_colors=10)
    g = sns.barplot(x = dataX, y = dataY, palette=pal)
    g.grid=False
    plt.xlabel('number of tweets')
    plt.ylabel('place')
    plt.title(title, fontweight='bold') 

    for i in range(10):
        g.text(dataX[i], i+0.22 ,round(data[i],3),color='black')
        
    plt.show()


# GraphSentimentDistribution(df_sen['sentiment'], 'Sentiment Distribution')
# GraphAxes(df_sen['sentiment'])
df_sen.describe()

'''
it's look like it is almost equally distributed , but the positive have a little bigger occurence here
Now let's take a look at the correlation between words that is included in the sentiment
'''
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

#print("Top Absolute Correlations")
#print(get_top_abs_correlations(df_sen, 10))

au = get_top_abs_correlations(df_sen, 15)
# print('Perfect Correlation')
# print(au[au==1])

'''
data to show graph 156 word that always occur together in every text,
although we dont really see how much of the sentences that is included there but the correlation is pretty high
'''
top10 = au[au<float(1)][0:20]
label = top10.index
label_list =[]
for i in label:
    for j in i:
        if(j not in label_list):
            label_list.append(j)
            
df_sen_corr = df_sen[label_list]
corr = df_sen_corr.corr()
for i in label_list:
    for j in label_list:
        if i!=j:
            corr[i][j] = round(corr[i][j],3)

'''
now let's take a look at other word, 
that is now always but often come together as the correlation quite high althoughh not equal to 1
'''
# GraphDataFeature(corr)

'''
most occured words among the set of words that is included in lexicon
'''
top15 = au[au<float(1)][0:15]
top15_word = df_sen.drop(['sentiment'],axis=1).sum().sort_values(ascending=False)[0:15]
# GraphTop15MostOftenOccuredWords(top15_word, top15_word.index, "Top 15 Most Often Occured Words")

df.isnull().sum()
# GraphNumericalFeature(df.corr(), np.triu(df.corr()))

'''
from the numerical feature, it seems the correlation is very low, now lets take a look at the others
'''
# GraphSentimentEveryLanguage(df,'lang','sentiment', 'Sentiment in every language used')

"""
it looks like people from this time frame that language 'ko' and 'und' 
always used to give positive sentiment while it is the opposite for 'pt' and 'es'
"""
cek_df = df.dropna(subset=['possibly_sensitive'])
cek_df = cek_df.reset_index(drop=True)
# GraphPossiblySensitive(cek_df,'possibly_sensitive','sentiment')

'''
the sensitive content does not indicating the statement as they are almost equally distributed among them
'''
df_place = df.groupby(['place']).mean().sort_values(by='sentiment',ascending=False)
df_place = df_place.reset_index()

df_place_dict = df.groupby(['place']).count().sort_values(by='id',ascending=False)['id'].to_dict()
df_place['number_of_tweets'] =  df_place.apply(lambda x:df_place_dict[x['place']],axis=1)

top10_place_pos = df_place.sort_values(by='sentiment',ascending=False)[0:10].reset_index(drop=True)
top10_place_neg = df_place.sort_values(by='sentiment',ascending=True)[0:10].reset_index(drop=True)
top10_place     = df_place.sort_values(by='number_of_tweets',ascending=False)[0:10].reset_index(drop=True)

'''
now let's take a look at some places where the sentiment that is made from them is tend to be sensitive and also for places that made otherwise
'''
# GraphMostPositiveNegativeSentimentPlace(top10_place_pos, top10_place_neg,'sentiment','place')

'''
last but not least, let's take a look at some places where tweets is most often come from
'''
# GraphTop10TweetPlace(top10_place['sentiment'], top10_place['number_of_tweets'], top10_place['place'], "Top 10 Number of Tweets place")

