# Problem Statment 

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

 

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

 

In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.

   1. Data sourcing and sentiment analysis
   2. Building a recommendation system
   3. Improving the recommendations using the sentiment analysis model
   4. Deploying the end-to-end project with a user interface

Steps involved in the project 
1. Exploratory data analysis
2. Data cleaning
3. Text preprocessing
4. Feature extraction 
4. Training the text classification model
5. Creating a recommedation systems (User based and Item Based choose the bestone)
6. Evaluating the model and recommedation system using the Test data 
7. Create flask application 
8. Deploy the application to heroku platform 


```python
pip freeze > requirement.txt
```


```python
#importing colab libraries
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
mydrive_path=''
```


```python
#importing libraries 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
%matplotlib inline



from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,f1_score,precision_score,accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
```


```python
pd.set_option('max_colwidth', 500)
```

# 1. Exploratory Analysis


```python
# Reading the input from folder 
master_df = pd.read_csv(mydrive_path+'sample30.csv')
df=master_df.copy()
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_userCity</th>
      <th>reviews_userProvince</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3148</th>
      <td>AVpe59io1cnluZ0-ZgDU</td>
      <td>Universal Home Video</td>
      <td>Movies, Music &amp; Books,Movies,Comedy,Movies &amp; TV Shows,Instawatch Movies By VUDU,Shop Instawatch,Movies &amp; TV,Ways To Shop Entertainment,Movies &amp; Tv On Blu-Ray,Movies &amp; Music,Instawatch,Blu-ray</td>
      <td>Universal</td>
      <td>My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)</td>
      <td>2017-01-07T00:00:00.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>Loved this sequel. Almost better than the first. Many laughs and feel-good moments.</td>
      <td>Great Sequel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>kammish</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4687</th>
      <td>AVpf0eb2LJeJML43EVSt</td>
      <td>Sony Pictures</td>
      <td>Movies, Music &amp; Books,Ways To Shop Entertainment,Movie &amp; Tv Box Sets,Movies,Horror,Movies &amp; TV Shows,All Horror,Movies &amp; Tv On Blu-Ray,Movies &amp; TV,Blu-ray,Action &amp; Adventure,Movies &amp; Music,Holiday Shop</td>
      <td>SONY CORP</td>
      <td>The Resident Evil Collection 5 Discs (blu-Ray)</td>
      <td>2017-03-17T00:00:00.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>4</td>
      <td>Bought the set for my gf since she is a fan of the resident evil series and for the price that we paid for, it was a great steal.</td>
      <td>Great deal for the set</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>keecha</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>8529</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2012-01-31T15:20:18.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>I use these wipes everyday! Can't clean without them!</td>
      <td>Great Product</td>
      <td>Delavan</td>
      <td>NaN</td>
      <td>debaaronj</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>16150</th>
      <td>AVpf63aJLJeJML43F__Q</td>
      <td>Burt's Bees</td>
      <td>Personal Care,Makeup,Lipstick, Lip Gloss, &amp; Lip Balm,Lip Gloss,Beauty,Lips,Beauty &amp; Personal Care,Skin Care,Lip Care,Lip Balms &amp; Treatments</td>
      <td>Burt's Bees</td>
      <td>Burt's Bees Lip Shimmer, Raisin</td>
      <td>2009-08-29T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>This lip shimmer smooths on so nicely and lasts. It also has a great peppermint taste. It is very easy to apply as well as, one does not need a lot to look great. My favorite is Fig.</td>
      <td>Wonderful lip shimmer</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>irishcow</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>8895</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2014-12-27T19:17:44.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>This product is very refreshing and is easy to use. This review was collected as part of a promotion.</td>
      <td>Great Product</td>
      <td>Houston</td>
      <td>NaN</td>
      <td>mrsmack</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Total reviews
total = len(df['reviews_text'])
print ("Number of reviews: ",total)

### How many unique reviewers?
print ("Number of unique reviewers: ",len(df['reviews_username'].unique()))
reviewer_prop = float(len(df['reviews_username'].unique())/total)
print ("Prop of unique reviewers: ",round(reviewer_prop,3))

### Average star score
print ("Average rating score: ",round(df['reviews_rating'].mean(),3))
```

    Number of reviews:  30000
    Number of unique reviewers:  24915
    Prop of unique reviewers:  0.831
    Average rating score:  4.483
    


```python
#data overivew
print('rows: ', df.shape[0])
print('columns: ', df.shape[1])
print('\nfeatures: ', df.columns.to_list())
print('\nmissing vlues: ', df.isnull().values.sum())
print('\nUnique values: \n', df.nunique())
```

    rows:  30000
    columns:  15
    
    features:  ['id', 'brand', 'categories', 'manufacturer', 'name', 'reviews_date', 'reviews_didPurchase', 'reviews_doRecommend', 'reviews_rating', 'reviews_text', 'reviews_title', 'reviews_userCity', 'reviews_userProvince', 'reviews_username', 'user_sentiment']
    
    missing vlues:  74980
    
    Unique values: 
     id                        271
    brand                     214
    categories                270
    manufacturer              227
    name                      271
    reviews_date             6857
    reviews_didPurchase         2
    reviews_doRecommend         2
    reviews_rating              5
    reviews_text            27282
    reviews_title           18535
    reviews_userCity          977
    reviews_userProvince       42
    reviews_username        24914
    user_sentiment              2
    dtype: int64
    


```python
# Info of the dataframe 
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 15 columns):
     #   Column                Non-Null Count  Dtype 
    ---  ------                --------------  ----- 
     0   id                    30000 non-null  object
     1   brand                 30000 non-null  object
     2   categories            30000 non-null  object
     3   manufacturer          29859 non-null  object
     4   name                  30000 non-null  object
     5   reviews_date          29954 non-null  object
     6   reviews_didPurchase   15932 non-null  object
     7   reviews_doRecommend   27430 non-null  object
     8   reviews_rating        30000 non-null  int64 
     9   reviews_text          30000 non-null  object
     10  reviews_title         29810 non-null  object
     11  reviews_userCity      1929 non-null   object
     12  reviews_userProvince  170 non-null    object
     13  reviews_username      29937 non-null  object
     14  user_sentiment        29999 non-null  object
    dtypes: int64(1), object(14)
    memory usage: 3.4+ MB
    


```python
# Number of occurences for each rating 
#plot ratings frequency
plt.figure(figsize=[10,5]) #[width, height]
x = list(df['reviews_rating'].value_counts().index)
y = list(df['reviews_rating'].value_counts())
plt.barh(x, y)
ticks_x = np.linspace(0, 50000, 6) # (start, end, no of ticks)
plt.xticks(ticks_x, fontsize=10, family='fantasy', color='black')
plt.yticks(size=15)

plt.title('Distribution of ratings', fontsize=20, weight='bold', color='navy', loc='center')
plt.xlabel('Count', fontsize=15, weight='bold', color='navy')
plt.ylabel('Ratings', fontsize=15, weight='bold', color='navy')
plt.legend(['reviews Rating'], shadow=True, loc=4)
```




    <matplotlib.legend.Legend at 0x1325f7d60>




    
![png](output_13_1.png)
    



```python
# Number of Postive and Negatives in the data frame showing the class imbalance
#Replace the Nan values to No Data for reviewers did purchase or not
df['reviews_didPurchase'].fillna('No Data', inplace=True)
#Distribution of reviews for actual purchasing customers
plt.figure(figsize=(10,8))
ax = sns.countplot(df['reviews_didPurchase'])
ax.set_xlabel(xlabel="Shoppers did purchase the product", fontsize=17)
ax.set_ylabel(ylabel='Count of Reviews', fontsize=17)
ax.axes.set_title('Number of Genuine Reviews', fontsize=17)
ax.tick_params(labelsize=13)
```

    /Users/jyoc/Library/Python/3.8/lib/python/site-packages/seaborn/_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_14_1.png)
    



```python
# To see any corrections are required in the dataframe is required using rating and user sentiment 
# from IPython.core.pylabtools import figsize
# figsize(10,10)
# sns.histplot(hue=df['reviews_rating'],x=df['user_sentiment'])
# plt.yticks(np.arange(0,30000,10000))
# plt.show()

plt.figure(figsize=(10,8))
ax = sns.histplot(hue=df['reviews_rating'],x=df['user_sentiment'])
ax.set_xlabel(xlabel="Shopper Sentiment", fontsize=17)
ax.set_ylabel(ylabel='Count of Reviews', fontsize=17)
ax.axes.set_title('Review Segregation', fontsize=17)
ax.tick_params(labelsize=13)
```


    
![png](output_15_0.png)
    



```python
df['user_sentiment'].value_counts()
```




    Positive    26632
    Negative     3367
    Name: user_sentiment, dtype: int64



* We need to correct the data available in the sentiments considering the rating of users


```python
# To download the stopwords from NLTK library
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /Users/jyoc/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
# To Check the most word occurence using word cloud
from wordcloud import WordCloud ,STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=300, max_font_size=40,
                     scale=3, random_state=1).generate(str(df['reviews_text'].value_counts()))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```


    
![png](output_19_0.png)
    


#  2. Data Cleaning


```python
# Finding the number of rows with Null values
df.isnull().sum()
```




    id                          0
    brand                       0
    categories                  0
    manufacturer              141
    name                        0
    reviews_date               46
    reviews_didPurchase         0
    reviews_doRecommend      2570
    reviews_rating              0
    reviews_text                0
    reviews_title             190
    reviews_userCity        28071
    reviews_userProvince    29830
    reviews_username           63
    user_sentiment              1
    dtype: int64




```python
#shape of the dataframe
df.shape
```




    (30000, 15)




```python
#From the null values percentages, columns reviews_userCity and reviews_userProvince can be dropped 
df = df.drop(columns=['reviews_userCity','reviews_userProvince'],axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AV13O1A8GV-KLJ3akUyj</td>
      <td>Universal Music</td>
      <td>Movies, Music &amp; Books,Music,R&amp;b,Movies &amp; TV,Movie Bundles &amp; Collections,CDs &amp; Vinyl,Rap &amp; Hip-Hop,Bass,Music on CD or Vinyl,Rap,Hip-Hop,Mainstream Rap,Pop Rap</td>
      <td>Universal Music Group / Cash Money</td>
      <td>Pink Friday: Roman Reloaded Re-Up (w/dvd)</td>
      <td>2012-11-30T06:21:45.000Z</td>
      <td>No Data</td>
      <td>NaN</td>
      <td>5</td>
      <td>i love this album. it's very good. more to the hip hop side than her current pop sound.. SO HYPE! i listen to this everyday at the gym! i give it 5star rating all the way. her metaphors are just crazy.</td>
      <td>Just Awesome</td>
      <td>joshua</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AV14LG0R-jtxr-f38QfS</td>
      <td>Lundberg</td>
      <td>Food,Packaged Foods,Snacks,Crackers,Snacks, Cookies &amp; Chips,Rice Cakes,Cakes</td>
      <td>Lundberg</td>
      <td>Lundberg Organic Cinnamon Toast Rice Cakes</td>
      <td>2017-07-09T00:00:00.000Z</td>
      <td>True</td>
      <td>NaN</td>
      <td>5</td>
      <td>Good flavor. This review was collected as part of a promotion.</td>
      <td>Good</td>
      <td>dorothy w</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AV14LG0R-jtxr-f38QfS</td>
      <td>Lundberg</td>
      <td>Food,Packaged Foods,Snacks,Crackers,Snacks, Cookies &amp; Chips,Rice Cakes,Cakes</td>
      <td>Lundberg</td>
      <td>Lundberg Organic Cinnamon Toast Rice Cakes</td>
      <td>2017-07-09T00:00:00.000Z</td>
      <td>True</td>
      <td>NaN</td>
      <td>5</td>
      <td>Good flavor.</td>
      <td>Good</td>
      <td>dorothy w</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AV16khLE-jtxr-f38VFn</td>
      <td>K-Y</td>
      <td>Personal Care,Medicine Cabinet,Lubricant/Spermicide,Health,Sexual Wellness,Lubricants</td>
      <td>K-Y</td>
      <td>K-Y Love Sensuality Pleasure Gel</td>
      <td>2016-01-06T00:00:00.000Z</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>I read through the reviews on here before looking in to buying one of the couples lubricants, and was ultimately disappointed that it didn't even live up to the reviews I had read. For starters, neither my boyfriend nor I could notice any sort of enhanced or 'captivating' sensation. What we did notice, however, was the messy consistency that was reminiscent of a more liquid-y vaseline. It was difficult to clean up, and was not a pleasant, especially since it lacked the 'captivating' sensatio...</td>
      <td>Disappointed</td>
      <td>rebecca</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AV16khLE-jtxr-f38VFn</td>
      <td>K-Y</td>
      <td>Personal Care,Medicine Cabinet,Lubricant/Spermicide,Health,Sexual Wellness,Lubricants</td>
      <td>K-Y</td>
      <td>K-Y Love Sensuality Pleasure Gel</td>
      <td>2016-12-21T00:00:00.000Z</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>My husband bought this gel for us. The gel caused irritation and it felt like it was burning my skin. I wouldn't recommend this gel.</td>
      <td>Irritation</td>
      <td>walker557</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Finding the number of rows with Null values
print("shape of the dataframe =",df.shape)
df.isnull().sum()/len(df)
```

    shape of the dataframe = (30000, 13)
    




    id                     0.000000
    brand                  0.000000
    categories             0.000000
    manufacturer           0.004700
    name                   0.000000
    reviews_date           0.001533
    reviews_didPurchase    0.000000
    reviews_doRecommend    0.085667
    reviews_rating         0.000000
    reviews_text           0.000000
    reviews_title          0.006333
    reviews_username       0.002100
    user_sentiment         0.000033
    dtype: float64




```python
# Before Updating the user sentiment columns
df['user_sentiment'].value_counts()
```




    Positive    26632
    Negative     3367
    Name: user_sentiment, dtype: int64




```python
# for correcting the user sentiment according to rating 
def review_sentiment_clear(x):
  if x >= 3 :
    return 'Postive'
  elif x > 0 and x < 3  :
    return 'Negative' 
```


```python
df['user_sentiment'] = df['reviews_rating'].apply(review_sentiment_clear)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AV13O1A8GV-KLJ3akUyj</td>
      <td>Universal Music</td>
      <td>Movies, Music &amp; Books,Music,R&amp;b,Movies &amp; TV,Movie Bundles &amp; Collections,CDs &amp; Vinyl,Rap &amp; Hip-Hop,Bass,Music on CD or Vinyl,Rap,Hip-Hop,Mainstream Rap,Pop Rap</td>
      <td>Universal Music Group / Cash Money</td>
      <td>Pink Friday: Roman Reloaded Re-Up (w/dvd)</td>
      <td>2012-11-30T06:21:45.000Z</td>
      <td>No Data</td>
      <td>NaN</td>
      <td>5</td>
      <td>i love this album. it's very good. more to the hip hop side than her current pop sound.. SO HYPE! i listen to this everyday at the gym! i give it 5star rating all the way. her metaphors are just crazy.</td>
      <td>Just Awesome</td>
      <td>joshua</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AV14LG0R-jtxr-f38QfS</td>
      <td>Lundberg</td>
      <td>Food,Packaged Foods,Snacks,Crackers,Snacks, Cookies &amp; Chips,Rice Cakes,Cakes</td>
      <td>Lundberg</td>
      <td>Lundberg Organic Cinnamon Toast Rice Cakes</td>
      <td>2017-07-09T00:00:00.000Z</td>
      <td>True</td>
      <td>NaN</td>
      <td>5</td>
      <td>Good flavor. This review was collected as part of a promotion.</td>
      <td>Good</td>
      <td>dorothy w</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AV14LG0R-jtxr-f38QfS</td>
      <td>Lundberg</td>
      <td>Food,Packaged Foods,Snacks,Crackers,Snacks, Cookies &amp; Chips,Rice Cakes,Cakes</td>
      <td>Lundberg</td>
      <td>Lundberg Organic Cinnamon Toast Rice Cakes</td>
      <td>2017-07-09T00:00:00.000Z</td>
      <td>True</td>
      <td>NaN</td>
      <td>5</td>
      <td>Good flavor.</td>
      <td>Good</td>
      <td>dorothy w</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AV16khLE-jtxr-f38VFn</td>
      <td>K-Y</td>
      <td>Personal Care,Medicine Cabinet,Lubricant/Spermicide,Health,Sexual Wellness,Lubricants</td>
      <td>K-Y</td>
      <td>K-Y Love Sensuality Pleasure Gel</td>
      <td>2016-01-06T00:00:00.000Z</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>I read through the reviews on here before looking in to buying one of the couples lubricants, and was ultimately disappointed that it didn't even live up to the reviews I had read. For starters, neither my boyfriend nor I could notice any sort of enhanced or 'captivating' sensation. What we did notice, however, was the messy consistency that was reminiscent of a more liquid-y vaseline. It was difficult to clean up, and was not a pleasant, especially since it lacked the 'captivating' sensatio...</td>
      <td>Disappointed</td>
      <td>rebecca</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AV16khLE-jtxr-f38VFn</td>
      <td>K-Y</td>
      <td>Personal Care,Medicine Cabinet,Lubricant/Spermicide,Health,Sexual Wellness,Lubricants</td>
      <td>K-Y</td>
      <td>K-Y Love Sensuality Pleasure Gel</td>
      <td>2016-12-21T00:00:00.000Z</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>My husband bought this gel for us. The gel caused irritation and it felt like it was burning my skin. I wouldn't recommend this gel.</td>
      <td>Irritation</td>
      <td>walker557</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# After corrections for user sentiment 
df['user_sentiment'].value_counts()
```




    Postive     28196
    Negative     1804
    Name: user_sentiment, dtype: int64



# 3. Text Preprocessing 

### Text lower cased , removed Special Charater and lemmatized


```python
#Common functions for cleaning the text data 
import nltk
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import html

# special_characters removal
def remove_special_characters(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

stopword_list= stopwords.words('english')

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas
```

    [nltk_data] Downloading package punkt to /Users/jyoc/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /Users/jyoc/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /Users/jyoc/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to /Users/jyoc/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!
    


```python
def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)
```


```python
# Take the Review comment and user sentiment as dataframe 
review_df = df[['reviews_text','user_sentiment']]
review_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviews_text</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i love this album. it's very good. more to the hip hop side than her current pop sound.. SO HYPE! i listen to this everyday at the gym! i give it 5star rating all the way. her metaphors are just crazy.</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good flavor. This review was collected as part of a promotion.</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Good flavor.</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I read through the reviews on here before looking in to buying one of the couples lubricants, and was ultimately disappointed that it didn't even live up to the reviews I had read. For starters, neither my boyfriend nor I could notice any sort of enhanced or 'captivating' sensation. What we did notice, however, was the messy consistency that was reminiscent of a more liquid-y vaseline. It was difficult to clean up, and was not a pleasant, especially since it lacked the 'captivating' sensatio...</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My husband bought this gel for us. The gel caused irritation and it felt like it was burning my skin. I wouldn't recommend this gel.</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a new column lemmatized_review using the emmatize_text function
review_df['lemmatized_text'] = review_df['reviews_text'].map(lambda text: normalize_and_lemmaize(text))
review_df.head()
```

    /var/folders/cs/8pdd31cj7lv6tjh615c3y3c00000gs/T/ipykernel_62540/3734498083.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviews_text</th>
      <th>user_sentiment</th>
      <th>lemmatized_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i love this album. it's very good. more to the hip hop side than her current pop sound.. SO HYPE! i listen to this everyday at the gym! i give it 5star rating all the way. her metaphors are just crazy.</td>
      <td>Postive</td>
      <td>love album good hip hop side current pop sound hype listen everyday gym give star rat way metaphors crazy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good flavor. This review was collected as part of a promotion.</td>
      <td>Postive</td>
      <td>good flavor review collect part promotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Good flavor.</td>
      <td>Postive</td>
      <td>good flavor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I read through the reviews on here before looking in to buying one of the couples lubricants, and was ultimately disappointed that it didn't even live up to the reviews I had read. For starters, neither my boyfriend nor I could notice any sort of enhanced or 'captivating' sensation. What we did notice, however, was the messy consistency that was reminiscent of a more liquid-y vaseline. It was difficult to clean up, and was not a pleasant, especially since it lacked the 'captivating' sensatio...</td>
      <td>Negative</td>
      <td>read review look buy one couple lubricants ultimately disappoint didnt even live review read starters neither boyfriend could notice sort enhance captivate sensation notice however messy consistency reminiscent liquidy vaseline difficult clean pleasant especially since lack captivate sensation expect im disappoint pay much lube wont use could use normal personal lubricant less money less mess</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My husband bought this gel for us. The gel caused irritation and it felt like it was burning my skin. I wouldn't recommend this gel.</td>
      <td>Negative</td>
      <td>husband buy gel us gel cause irritation felt like burn skin wouldnt recommend gel</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new dataframe with lemmatized text and user sentiment 
review_new_df = review_df[['lemmatized_text','user_sentiment']]
review_new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lemmatized_text</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love album good hip hop side current pop sound hype listen everyday gym give star rat way metaphors crazy</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>good flavor review collect part promotion</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>good flavor</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>read review look buy one couple lubricants ultimately disappoint didnt even live review read starters neither boyfriend could notice sort enhance captivate sensation notice however messy consistency reminiscent liquidy vaseline difficult clean pleasant especially since lack captivate sensation expect im disappoint pay much lube wont use could use normal personal lubricant less money less mess</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>husband buy gel us gel cause irritation felt like burn skin wouldnt recommend gel</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29995</th>
      <td>get conditioner influenster try im love far oily hair use end hair feel amaze soft mess review collect part promotion</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>29996</th>
      <td>love receive review purpose influenster leave hair feel fresh smell great</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>29997</th>
      <td>first love smell product wash hair smooth easy brush receive product influenster test purpose opinions review collect part promotion</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>29998</th>
      <td>receive influenster never go back anything else normally dont use conditioner hair oily fine make hair feel heavy doesnt get oily day really fantastic plan buy future review collect part promotion</td>
      <td>Postive</td>
    </tr>
    <tr>
      <th>29999</th>
      <td>receive product complimentary influenster really save hair product really give extra boost health strength bring hair back life hasnt help hair many ways review collect part promotion</td>
      <td>Postive</td>
    </tr>
  </tbody>
</table>
<p>30000 rows × 2 columns</p>
</div>




```python
#Encode the negative and postive to 0 and 1 respectively 
review_new_df['user_sentiment'] = review_new_df['user_sentiment'].map({'Negative':0,'Postive':1})
review_new_df.head()
```

    /var/folders/cs/8pdd31cj7lv6tjh615c3y3c00000gs/T/ipykernel_62540/125465557.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lemmatized_text</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love album good hip hop side current pop sound hype listen everyday gym give star rat way metaphors crazy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>good flavor review collect part promotion</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>good flavor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>read review look buy one couple lubricants ultimately disappoint didnt even live review read starters neither boyfriend could notice sort enhance captivate sensation notice however messy consistency reminiscent liquidy vaseline difficult clean pleasant especially since lack captivate sensation expect im disappoint pay much lube wont use could use normal personal lubricant less money less mess</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>husband buy gel us gel cause irritation felt like burn skin wouldnt recommend gel</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Dividing the dataset into train and test data and handle the class imbalance


```python
from collections import Counter
from imblearn.over_sampling import SMOTE
```


```python
# Train and Test Divide
x_train,x_test,y_train,y_test = train_test_split(review_new_df['lemmatized_text'],review_new_df['user_sentiment'],train_size=0.75,random_state=45,stratify=review_new_df['user_sentiment'])
y_train.value_counts()
```




    1    21147
    0     1353
    Name: user_sentiment, dtype: int64



# 4. Feature Extraction using Count Vectorizer and TFIDF Transformer 


```python
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer()
x_count = count_vect.fit_transform(x_train)


tfidf_transformer = TfidfTransformer()
x_train_transformed = tfidf_transformer.fit_transform(x_count)
x_train_transformed.shape
```




    (22500, 14711)




```python
#creating the pickle for countvectorizer and TFIDF Transformer
import pickle
pickle.dump(count_vect,open(mydrive_path+'pickle_file/count_vector.pkl','wb'))
pickle.dump(tfidf_transformer,open(mydrive_path+'pickle_file/tfidf_transformer.pkl','wb'))
```


```python
count = Counter(y_train)
print('Before sampling :',count)

sampler = SMOTE()

x_train_sm,y_train_sm = sampler.fit_resample(x_train_transformed,y_train)

count = Counter(y_train_sm)
print('After sampling :',count)
```

    Before sampling : Counter({1: 21147, 0: 1353})
    After sampling : Counter({1: 21147, 0: 21147})
    

# 5. Training text classification model
- Logistic Regression
- Random Forest Classifer
- XGBoost
#### Choose the best model with hyperparameter tuning 


```python
# Function for Metrics
performance=[]

def model_metrics(y,y_pred,model_name,metrics):
  Accuracy = accuracy_score(y,y_pred)
  roc = roc_auc_score(y,y_pred)
  confusion = confusion_matrix(y,y_pred)
  precision = precision_score(y,y_pred)
  f1 = f1_score(y,y_pred)
  TP = confusion[1,1]  # true positive
  TN = confusion[0,0]  # true negatives
  FP = confusion[0,1]  # false positives
  FN = confusion[1,0]  # false negatives
  sensitivity= TP / float(TP+FN)
  specificity = TN / float(TN+FP)

  print("*"*50)
  print('Confusion Matrix =')
  print(confusion)
  print("sensitivity of the %s = %f" % (model_name,round(sensitivity,2)))
  print("specificity of the %s = %f" % (model_name,round(specificity,2)))
  print("Accuracy Score of %s = %f" % (model_name,Accuracy))
  print('ROC AUC score of %s = %f' % (model_name,roc))
  print("Report=",)
  print(classification_report(y,y_pred))
  print("*"*50)
  metrics.append(dict({'Model_name':model_name,
                       'Accuracy':Accuracy,
                       'Roc_auc_score':roc,
                       'Precision':precision,
                       'F1_score':f1}))
  return metrics


```

## Logistic Regression


```python
# 1. Logsitic Regression 
lr = LogisticRegression()
lr.fit(x_train_sm,y_train_sm)
```




    LogisticRegression()




```python
y_pred = lr.predict(x_train_sm)
peformance = model_metrics(y_train_sm,y_pred,'Logistic Regression',performance)
```

    **************************************************
    Confusion Matrix =
    [[20834   313]
     [  617 20530]]
    sensitivity of the Logistic Regression = 0.970000
    specificity of the Logistic Regression = 0.990000
    Accuracy Score of Logistic Regression = 0.978011
    ROC AUC score of Logistic Regression = 0.978011
    Report=
                  precision    recall  f1-score   support
    
               0       0.97      0.99      0.98     21147
               1       0.98      0.97      0.98     21147
    
        accuracy                           0.98     42294
       macro avg       0.98      0.98      0.98     42294
    weighted avg       0.98      0.98      0.98     42294
    
    **************************************************
    

## RandomForest Classifier


```python
# 2. RandomForest Classifier
rf = RandomForestClassifier()
rf.fit(x_train_sm,y_train_sm)
```




    RandomForestClassifier()




```python
y_pred_rf = rf.predict(x_train_sm)
performance = model_metrics(y_train_sm,y_pred_rf,'RandomForestClassifier',performance)
```

    **************************************************
    Confusion Matrix =
    [[21145     2]
     [    1 21146]]
    sensitivity of the RandomForestClassifier = 1.000000
    specificity of the RandomForestClassifier = 1.000000
    Accuracy Score of RandomForestClassifier = 0.999929
    ROC AUC score of RandomForestClassifier = 0.999929
    Report=
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     21147
               1       1.00      1.00      1.00     21147
    
        accuracy                           1.00     42294
       macro avg       1.00      1.00      1.00     42294
    weighted avg       1.00      1.00      1.00     42294
    
    **************************************************
    

## AdaBoost Classifier


```python
xgba = GradientBoostingClassifier()
xgba.fit(x_train_sm,y_train_sm)
y_pred_xgb = xgba.predict(x_train_sm)
peformance = model_metrics(y_train_sm,y_pred_rf,'AdaBoostclassifier',peformance)


```

    **************************************************
    Confusion Matrix =
    [[21145     2]
     [    1 21146]]
    sensitivity of the AdaBoostclassifier = 1.000000
    specificity of the AdaBoostclassifier = 1.000000
    Accuracy Score of AdaBoostclassifier = 0.999929
    ROC AUC score of AdaBoostclassifier = 0.999929
    Report=
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     21147
               1       1.00      1.00      1.00     21147
    
        accuracy                           1.00     42294
       macro avg       1.00      1.00      1.00     42294
    weighted avg       1.00      1.00      1.00     42294
    
    **************************************************
    

## XGBoost


```python
import xgboost as xgb
```


```python
#4.XGBoostClassifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(x_train_sm,y_train_sm)
y_pred_xgbc = xgb_classifier.predict(x_train_sm)
peformance = model_metrics(y_train_sm,y_pred_xgbc,'XGBClassifier',peformance)

```

    /Users/jyoc/Library/Python/3.8/lib/python/site-packages/xgboost/sklearn.py:1224: UserWarning:
    
    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    
    

    [00:55:42] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    **************************************************
    Confusion Matrix =
    [[20844   303]
     [  105 21042]]
    sensitivity of the XGBClassifier = 1.000000
    specificity of the XGBClassifier = 0.990000
    Accuracy Score of XGBClassifier = 0.990353
    ROC AUC score of XGBClassifier = 0.990353
    Report=
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99     21147
               1       0.99      1.00      0.99     21147
    
        accuracy                           0.99     42294
       macro avg       0.99      0.99      0.99     42294
    weighted avg       0.99      0.99      0.99     42294
    
    **************************************************
    


```python
metrics_df = pd.DataFrame(performance)
metrics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model_name</th>
      <th>Accuracy</th>
      <th>Roc_auc_score</th>
      <th>Precision</th>
      <th>F1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.978011</td>
      <td>0.978011</td>
      <td>0.984983</td>
      <td>0.977852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.999929</td>
      <td>0.999929</td>
      <td>0.999905</td>
      <td>0.999929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoostclassifier</td>
      <td>0.999929</td>
      <td>0.999929</td>
      <td>0.999905</td>
      <td>0.999929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBClassifier</td>
      <td>0.990353</td>
      <td>0.990353</td>
      <td>0.985805</td>
      <td>0.990398</td>
    </tr>
  </tbody>
</table>
</div>



## Hyperparameter Tuning of models 


```python
n_estimators = [200,400,600]
max_depth = [6,10,15]
min_samples_leaf = [5,6,8]
criterion  = ['gini','entropy']
params = {'n_estimators':n_estimators,
          'max_depth':max_depth,
          'min_samples_leaf': min_samples_leaf,
          'criterion':criterion}
```


```python
grid_cv = GridSearchCV(estimator=rf,
                       param_grid=params,
                       n_jobs = -1,
                       scoring = 'roc_auc',
                       verbose = 1)
```


```python
grid_cv.fit(x_train_sm,y_train_sm)
```

    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    




    GridSearchCV(estimator=RandomForestClassifier(), n_jobs=-1,
                 param_grid={'criterion': ['gini', 'entropy'],
                             'max_depth': [6, 10, 15],
                             'min_samples_leaf': [5, 6, 8],
                             'n_estimators': [200, 400, 600]},
                 scoring='roc_auc', verbose=1)




```python
rf_final=grid_cv.best_estimator_
rf_final
```




    RandomForestClassifier(max_depth=15, min_samples_leaf=5, n_estimators=600)




```python
pickle.dump(rf_final,open(mydrive_path+'pickle_file/RandomForest_classifier.pkl','wb'))
```


```python
grid_cv.best_score_
```




    0.9780470397236973




```python
y_pred_rfgcv = rf_final.predict(x_train_sm)
performance = model_metrics(y_train_sm,y_pred_rfgcv,'RandomForestClassifier with hyperparmater',performance)
```

    **************************************************
    Confusion Matrix =
    [[17875  3272]
     [  631 20516]]
    sensitivity of the RandomForestClassifier with hyperparmater = 0.970000
    specificity of the RandomForestClassifier with hyperparmater = 0.850000
    Accuracy Score of RandomForestClassifier with hyperparmater = 0.907717
    ROC AUC score of RandomForestClassifier with hyperparmater = 0.907717
    Report=
                  precision    recall  f1-score   support
    
               0       0.97      0.85      0.90     21147
               1       0.86      0.97      0.91     21147
    
        accuracy                           0.91     42294
       macro avg       0.91      0.91      0.91     42294
    weighted avg       0.91      0.91      0.91     42294
    
    **************************************************
    


```python
n_estimators = [200,400,600]
params_1 = {'n_estimators':n_estimators}   
```


```python
grid_cv_boost = GridSearchCV(estimator=xgba,
                       param_grid=params_1,
                       n_jobs = -1,
                       scoring = 'roc_auc',
                       verbose = 1)
```


```python
grid_cv_boost.fit(x_train_sm,y_train_sm)
print('Best score for GradientBoosting=',grid_cv_boost.best_score_)

```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    Best score for GradientBoosting= 0.9954032773019563
    


```python
xgb_final=grid_cv_boost.best_estimator_
xgb_final
```




    GradientBoostingClassifier(n_estimators=600)




```python
y_pred_xgbgcv = xgb_final.predict(x_train_sm)
peformance = model_metrics(y_train_sm,y_pred_xgbgcv,'GradientBoostClassifier with n = 600',peformance)
```

    **************************************************
    Confusion Matrix =
    [[20749   398]
     [  329 20818]]
    sensitivity of the GradientBoostClassifier with n = 600 = 0.980000
    specificity of the GradientBoostClassifier with n = 600 = 0.980000
    Accuracy Score of GradientBoostClassifier with n = 600 = 0.982811
    ROC AUC score of GradientBoostClassifier with n = 600 = 0.982811
    Report=
                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98     21147
               1       0.98      0.98      0.98     21147
    
        accuracy                           0.98     42294
       macro avg       0.98      0.98      0.98     42294
    weighted avg       0.98      0.98      0.98     42294
    
    **************************************************
    


```python
max_depth = [5,6,7,10]

params_2 = {'max_depth':max_depth
}
grid_cv_boost2 = GridSearchCV(estimator=xgb_final,
                       param_grid=params_2,
                       n_jobs = -1,
                       scoring = 'roc_auc',
                       verbose = 1)
```


```python
grid_cv_boost2.fit(x_train_sm,y_train_sm)
print('Best score for GradientBoosting=',grid_cv_boost2.best_score_)
grid_cv_boost2.best_estimator_
```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    Best score for GradientBoosting= 0.9984064786121356
    




    GradientBoostingClassifier(max_depth=10, n_estimators=600)




```python
min_samples_split = [10,20,30]
params_2 = {'min_samples_split': min_samples_split
}
grid_cv_boost3 = GridSearchCV(estimator=grid_cv_boost2.best_estimator_,
                       param_grid=params_2,
                       n_jobs = -1,
                       scoring = 'roc_auc',
                       verbose = 1)

grid_cv_boost3.fit(x_train_sm,y_train_sm)
print('Best score for GradientBoosting=',grid_cv_boost3.best_score_)
grid_cv_boost3.best_estimator_
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    Best score for GradientBoosting= 0.9984306001799554
    




    GradientBoostingClassifier(max_depth=10, min_samples_split=10, n_estimators=600)




```python
y_pred_xgbgcv2 = grid_cv_boost3.best_estimator_.predict(x_train_sm)
peformance = model_metrics(y_train_sm,y_pred_xgbgcv2,'GradientBoostClassifier with param2',peformance)
performance
```

    **************************************************
    Confusion Matrix =
    [[21140     7]
     [    0 21147]]
    sensitivity of the GradientBoostClassifier with param2 = 1.000000
    specificity of the GradientBoostClassifier with param2 = 1.000000
    Accuracy Score of GradientBoostClassifier with param2 = 0.999834
    ROC AUC score of GradientBoostClassifier with param2 = 0.999834
    Report=
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     21147
               1       1.00      1.00      1.00     21147
    
        accuracy                           1.00     42294
       macro avg       1.00      1.00      1.00     42294
    weighted avg       1.00      1.00      1.00     42294
    
    **************************************************
    




    [{'Model_name': 'Logistic Regression',
      'Accuracy': 0.9780110653993475,
      'Roc_auc_score': 0.9780110653993476,
      'Precision': 0.9849829679028931,
      'F1_score': 0.9778518694927364},
     {'Model_name': 'RandomForestClassifier',
      'Accuracy': 0.9999290679529012,
      'Roc_auc_score': 0.9999290679529012,
      'Precision': 0.9999054284093059,
      'F1_score': 0.9999290696299798},
     {'Model_name': 'AdaBoostclassifier',
      'Accuracy': 0.9999290679529012,
      'Roc_auc_score': 0.9999290679529012,
      'Precision': 0.9999054284093059,
      'F1_score': 0.9999290696299798},
     {'Model_name': 'XGBClassifier',
      'Accuracy': 0.9903532415945524,
      'Roc_auc_score': 0.9903532415945524,
      'Precision': 0.9858046380885453,
      'F1_score': 0.9903981926009603},
     {'Model_name': 'RandomForestClassifier with hyperparmater',
      'Accuracy': 0.9077174067243581,
      'Roc_auc_score': 0.9077174067243581,
      'Precision': 0.8624516562972927,
      'F1_score': 0.9131412039612773},
     {'Model_name': 'GradientBoostClassifier with n = 600',
      'Accuracy': 0.9828108005863716,
      'Roc_auc_score': 0.9828108005863715,
      'Precision': 0.9812405731523378,
      'F1_score': 0.9828387980076954},
     {'Model_name': 'GradientBoostClassifier with param2',
      'Accuracy': 0.9998344918901027,
      'Roc_auc_score': 0.9998344918901025,
      'Precision': 0.999669093315685,
      'F1_score': 0.9998345192785041}]




```python
metrics_df = pd.DataFrame(performance)
metrics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model_name</th>
      <th>Accuracy</th>
      <th>Roc_auc_score</th>
      <th>Precision</th>
      <th>F1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.978011</td>
      <td>0.978011</td>
      <td>0.984983</td>
      <td>0.977852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.999929</td>
      <td>0.999929</td>
      <td>0.999905</td>
      <td>0.999929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoostclassifier</td>
      <td>0.999929</td>
      <td>0.999929</td>
      <td>0.999905</td>
      <td>0.999929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBClassifier</td>
      <td>0.990353</td>
      <td>0.990353</td>
      <td>0.985805</td>
      <td>0.990398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomForestClassifier with hyperparmater</td>
      <td>0.907717</td>
      <td>0.907717</td>
      <td>0.862452</td>
      <td>0.913141</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GradientBoostClassifier with n = 600</td>
      <td>0.982811</td>
      <td>0.982811</td>
      <td>0.981241</td>
      <td>0.982839</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GradientBoostClassifier with param2</td>
      <td>0.999834</td>
      <td>0.999834</td>
      <td>0.999669</td>
      <td>0.999835</td>
    </tr>
  </tbody>
</table>
</div>




```python
rf_final = pickle.load(open(mydrive_path+'pickle_file/RandomForest_classifier.pkl','rb'))
```


```python
# After doing multiple tuning we get the below model and will be used in the sentiment based analysis
final_model = GradientBoostingClassifier(max_depth=10, min_samples_split=20, n_estimators=600)
```


```python
final_model.fit(x_train_sm,y_train_sm)
```




    GradientBoostingClassifier(max_depth=10, min_samples_split=20, n_estimators=600)




```python
pickle.dump(final_model,open(mydrive_path+'pickle_file/final_model.pkl','wb'))
```


```python
#Evaluatopn between lr , rf and boost 
test_performance=[]
test_word_vect = count_vect.transform(x_test)
test_tfidf_vect = tfidf_transformer.transform(test_word_vect)

y_test_pred_lr = lr.predict(test_tfidf_vect)
test_peformance = model_metrics(y_test,y_test_pred_lr,'Logistic Regression',test_performance)

y_test_pred_xgbc = xgb_classifier.predict(test_tfidf_vect)
test_peformance = model_metrics(y_test,y_test_pred_xgbc,'XGBoost Classifier',test_performance)

y_test_pred_rf = rf_final.predict(test_tfidf_vect)
test_peformance = model_metrics(y_test,y_test_pred_rf,'Tuned RandomForestClassifier',test_performance)

y_test_pred_xgb = final_model.predict(test_tfidf_vect)
test_peformance = model_metrics(y_test,y_test_pred_xgb,'Tuned GradientBoostClassifier',test_performance)

test_metrics_df = pd.DataFrame(test_performance)
test_metrics_df
```

    **************************************************
    Confusion Matrix =
    [[ 348  103]
     [ 262 6787]]
    sensitivity of the Logistic Regression = 0.960000
    specificity of the Logistic Regression = 0.770000
    Accuracy Score of Logistic Regression = 0.951333
    ROC AUC score of Logistic Regression = 0.867225
    Report=
                  precision    recall  f1-score   support
    
               0       0.57      0.77      0.66       451
               1       0.99      0.96      0.97      7049
    
        accuracy                           0.95      7500
       macro avg       0.78      0.87      0.81      7500
    weighted avg       0.96      0.95      0.95      7500
    
    **************************************************
    **************************************************
    Confusion Matrix =
    [[ 294  157]
     [ 101 6948]]
    sensitivity of the XGBoost Classifier = 0.990000
    specificity of the XGBoost Classifier = 0.650000
    Accuracy Score of XGBoost Classifier = 0.965600
    ROC AUC score of XGBoost Classifier = 0.818778
    Report=
                  precision    recall  f1-score   support
    
               0       0.74      0.65      0.70       451
               1       0.98      0.99      0.98      7049
    
        accuracy                           0.97      7500
       macro avg       0.86      0.82      0.84      7500
    weighted avg       0.96      0.97      0.96      7500
    
    **************************************************
    **************************************************
    Confusion Matrix =
    [[ 270  181]
     [ 229 6820]]
    sensitivity of the Tuned RandomForestClassifier = 0.970000
    specificity of the Tuned RandomForestClassifier = 0.600000
    Accuracy Score of Tuned RandomForestClassifier = 0.945333
    ROC AUC score of Tuned RandomForestClassifier = 0.783091
    Report=
                  precision    recall  f1-score   support
    
               0       0.54      0.60      0.57       451
               1       0.97      0.97      0.97      7049
    
        accuracy                           0.95      7500
       macro avg       0.76      0.78      0.77      7500
    weighted avg       0.95      0.95      0.95      7500
    
    **************************************************
    **************************************************
    Confusion Matrix =
    [[ 298  153]
     [  71 6978]]
    sensitivity of the Tuned GradientBoostClassifier = 0.990000
    specificity of the Tuned GradientBoostClassifier = 0.660000
    Accuracy Score of Tuned GradientBoostClassifier = 0.970133
    ROC AUC score of Tuned GradientBoostClassifier = 0.825341
    Report=
                  precision    recall  f1-score   support
    
               0       0.81      0.66      0.73       451
               1       0.98      0.99      0.98      7049
    
        accuracy                           0.97      7500
       macro avg       0.89      0.83      0.86      7500
    weighted avg       0.97      0.97      0.97      7500
    
    **************************************************
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model_name</th>
      <th>Accuracy</th>
      <th>Roc_auc_score</th>
      <th>Precision</th>
      <th>F1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.951333</td>
      <td>0.867225</td>
      <td>0.985051</td>
      <td>0.973814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>XGBoost Classifier</td>
      <td>0.965600</td>
      <td>0.818778</td>
      <td>0.977903</td>
      <td>0.981772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tuned RandomForestClassifier</td>
      <td>0.945333</td>
      <td>0.783091</td>
      <td>0.974147</td>
      <td>0.970819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tuned GradientBoostClassifier</td>
      <td>0.970133</td>
      <td>0.825341</td>
      <td>0.978544</td>
      <td>0.984203</td>
    </tr>
  </tbody>
</table>
</div>



### Evaluation with test data after comparing 
- Considering roc_auc_score ,performance. 
- Logistic Regression is having more score and have good accuracy 


```python
pickle.dump(lr,open(mydrive_path+'pickle_file/model.pkl','wb'))
```

# 5. Recommedation system
- User and User recommedation system 
- Item and Item recommedation system 

## User and User recommedation 



```python
df = pd.read_csv(mydrive_path+'sample30.csv')
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_userCity</th>
      <th>reviews_userProvince</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19140</th>
      <td>AVpfJP1C1cnluZ0-e3Xy</td>
      <td>Clorox</td>
      <td>Household Chemicals,Household Cleaners,Bath &amp; Shower Cleaner,Household Essentials,Cleaning Supplies,Bathroom Cleaners,Prime Pantry,Bathroom,Featured Brands,Home And Storage &amp; Org,Clorox,All-purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Target Restock,Food &amp; Grocery</td>
      <td>AmazonUs/CLOO7</td>
      <td>Clorox Disinfecting Bathroom Cleaner</td>
      <td>2012-01-26T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>I use Clorox Wipes everywhere in my home!!! I couldn't live without them. thank you!!!</td>
      <td>Amazing!</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lboogs</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>10872</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2016-09-16T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>Buy for the office ,school and home gavebto have it all year! This review was collected as part of a promotion.</td>
      <td>Great on everything</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>linboo</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>22567</th>
      <td>AVpfOmKwLJeJML435GM7</td>
      <td>Clear Scalp &amp; Hair Therapy</td>
      <td>Personal Care,Hair Care,Shampoo,Featured Brands,Health &amp; Beauty,Unilever,Beauty,Shampoo &amp; Conditioner,Shampoos,Hair Care &amp; Styling,Shampoos &amp; Conditioners,Ways To Shop</td>
      <td>Clear</td>
      <td>Clear Scalp &amp; Hair Therapy Total Care Nourishing Shampoo</td>
      <td>2016-02-10T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>4</td>
      <td>The shampoo lathers up nice and does a good job of cleaning my scalp and hair. I'm even able to run my hands through my hair while shampooing. It feels conditioned and not drying out my hair. This review was collected as part of a promotion.</td>
      <td>Great lather.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mbrunello1</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3238</th>
      <td>AVpe59io1cnluZ0-ZgDU</td>
      <td>Universal Home Video</td>
      <td>Movies, Music &amp; Books,Movies,Comedy,Movies &amp; TV Shows,Instawatch Movies By VUDU,Shop Instawatch,Movies &amp; TV,Ways To Shop Entertainment,Movies &amp; Tv On Blu-Ray,Movies &amp; Music,Instawatch,Blu-ray</td>
      <td>Universal</td>
      <td>My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)</td>
      <td>2017-05-18T00:00:00.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>Really good movie for family watching. Funny and engaging.</td>
      <td>Great family movie</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>castle</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7978</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2012-01-26T22:47:42.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>I love to clean around the house with Clorox disinfecting wipes! They make my life a lot easier!</td>
      <td>I Love This Product!</td>
      <td>San Jose</td>
      <td>NaN</td>
      <td>marii</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df['name'].unique())
```




    271




```python
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,train_size=0.70,random_state=45)
print('train shape = ',train.shape)
print('test shape = ',test.shape)
```

    train shape =  (21000, 15)
    test shape =  (9000, 15)
    


```python
#using train dataset and create correlation matrix 
train_pivot = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',data=train).fillna(1)
train_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>5302050 15/16 FCT/HOSE ADAPTOR</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>...</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Walkers Stem Ginger Shortbread</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00sab00</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>02dakota</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>02deuce</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0325home</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>06stidriver</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>zuttle</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>zwithanx</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>zxcsdfd</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>zyiah4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>zzdiane</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>18205 rows × 254 columns</p>
</div>




```python
#Creating the train and test dataset for predicting and evaluating the correlation
#fill 1 in place of Nan for prediction 
train_pivot1 = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',data=train).fillna(1)
```


```python
train_pivot1.loc['piggyboy420']
```




    name
    0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest           1.0
    100:Complete First Season (blu-Ray)                                     1.0
    2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black       1.0
    2x Ultra Era with Oxi Booster, 50fl oz                                  1.0
    42 Dual Drop Leaf Table with 2 Madrid Chairs"                           1.0
                                                                           ... 
    WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black    1.0
    Weleda Everon Lip Balm                                                  1.0
    Windex Original Glass Cleaner Refill 67.6oz (2 Liter)                   1.0
    Yes To Carrots Nourishing Body Wash                                     1.0
    Yes To Grapefruit Rejuvenating Body Wash                                1.0
    Name: piggyboy420, Length: 254, dtype: float64




```python
# here we are going use the adjusted cosine similarity 
import numpy as np

def cosine_similarity(df):
    # using the adjusted cosine similarity 
    mean_df = np.nanmean(df,axis=1)
    substracted_df = (df.T - mean_df).T # Normalized dataset
    # using the pairwise_distance for cosine similarity 
    user_correlation = 1- pairwise_distances (substracted_df.fillna(0),metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    return user_correlation,substracted_df
    
```


```python
user_corr_matrix,normalized_df = cosine_similarity(train_pivot1)
user_corr_matrix
```




    array([[ 1.        , -0.00395257, -0.00395257, ..., -0.00395257,
            -0.00395257,  1.        ],
           [-0.00395257,  1.        ,  1.        , ..., -0.00395257,
            -0.00395257, -0.00395257],
           [-0.00395257,  1.        ,  1.        , ..., -0.00395257,
            -0.00395257, -0.00395257],
           ...,
           [-0.00395257, -0.00395257, -0.00395257, ...,  1.        ,
             1.        , -0.00395257],
           [-0.00395257, -0.00395257, -0.00395257, ...,  1.        ,
             1.        , -0.00395257],
           [ 1.        , -0.00395257, -0.00395257, ..., -0.00395257,
            -0.00395257,  1.        ]])




```python
user_corr_matrix.shape
```




    (18205, 18205)




```python
user_corr_matrix[user_corr_matrix < 0] = 0
user_corr_matrix.shape
```




    (18205, 18205)




```python
df[df['reviews_username'] == 'zzz1127']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_userCity</th>
      <th>reviews_userProvince</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7256</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2014-12-03T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>4</td>
      <td>These wipes are very handy for getting your cleaning done quickly. I keep them in the bathroom and use them to wipe down all surface areas for a quick 10 minute cleaning. Keep it up every week and it's that easy to maintain a clean room. This review was collected as part of a promotion.</td>
      <td>Handy Wipes for Quick Cleaning</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>zzz1127</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_pred_ratings = np.dot(user_corr_matrix,train_pivot1.fillna(0))
user_pred_ratings
```




    array([[ 437.75576386,  438.26995035,  437.75576386, ...,  443.40142748,
             440.05601875,  437.75576386],
           [2120.11694472, 2138.77377911, 2120.11694472, ..., 2132.78849166,
            2120.11694472, 2121.98528539],
           [2120.11694472, 2138.77377911, 2120.11694472, ..., 2132.78849166,
            2120.11694472, 2121.98528539],
           ...,
           [5461.34652523, 5465.56431548, 5461.34652523, ..., 5467.89333212,
            5464.16935704, 5461.34652523],
           [5461.34652523, 5465.56431548, 5461.34652523, ..., 5467.89333212,
            5464.16935704, 5461.34652523],
           [ 437.75576386,  438.26995035,  437.75576386, ...,  443.40142748,
             440.05601875,  437.75576386]])




```python
user_pred_ratings.shape
```




    (18205, 254)




```python
user_final_rating = np.multiply(user_pred_ratings,train_pivot)
user_final_rating
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>5302050 15/16 FCT/HOSE ADAPTOR</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>...</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Walkers Stem Ginger Shortbread</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00sab00</th>
      <td>437.755764</td>
      <td>438.269950</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>440.056019</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>441.303557</td>
      <td>...</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>438.873393</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>443.401427</td>
      <td>440.056019</td>
      <td>437.755764</td>
    </tr>
    <tr>
      <th>02dakota</th>
      <td>2120.116945</td>
      <td>2138.773779</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2133.045416</td>
      <td>...</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2121.985285</td>
      <td>2120.116945</td>
      <td>2122.191680</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2132.788492</td>
      <td>2120.116945</td>
      <td>2121.985285</td>
    </tr>
    <tr>
      <th>02deuce</th>
      <td>2120.116945</td>
      <td>2138.773779</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2133.045416</td>
      <td>...</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2121.985285</td>
      <td>2120.116945</td>
      <td>2122.191680</td>
      <td>2120.116945</td>
      <td>2120.116945</td>
      <td>2132.788492</td>
      <td>2120.116945</td>
      <td>2121.985285</td>
    </tr>
    <tr>
      <th>0325home</th>
      <td>5461.346525</td>
      <td>5465.564315</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5465.138556</td>
      <td>...</td>
      <td>5469.987314</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5467.893332</td>
      <td>5464.169357</td>
      <td>5461.346525</td>
    </tr>
    <tr>
      <th>06stidriver</th>
      <td>5461.346525</td>
      <td>5465.564315</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5465.138556</td>
      <td>...</td>
      <td>5469.987314</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5467.893332</td>
      <td>5464.169357</td>
      <td>5461.346525</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>zuttle</th>
      <td>456.106425</td>
      <td>458.597546</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>461.356739</td>
      <td>...</td>
      <td>457.883951</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>456.106425</td>
      <td>459.712962</td>
      <td>456.106425</td>
      <td>456.106425</td>
    </tr>
    <tr>
      <th>zwithanx</th>
      <td>5461.346525</td>
      <td>5465.564315</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5465.138556</td>
      <td>...</td>
      <td>5469.987314</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5467.893332</td>
      <td>5464.169357</td>
      <td>5461.346525</td>
    </tr>
    <tr>
      <th>zxcsdfd</th>
      <td>5461.346525</td>
      <td>5465.564315</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5465.138556</td>
      <td>...</td>
      <td>5469.987314</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5467.893332</td>
      <td>5464.169357</td>
      <td>5461.346525</td>
    </tr>
    <tr>
      <th>zyiah4</th>
      <td>5461.346525</td>
      <td>5465.564315</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5465.138556</td>
      <td>...</td>
      <td>5469.987314</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5461.346525</td>
      <td>5467.893332</td>
      <td>5464.169357</td>
      <td>5461.346525</td>
    </tr>
    <tr>
      <th>zzdiane</th>
      <td>437.755764</td>
      <td>438.269950</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>440.056019</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>441.303557</td>
      <td>...</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>438.873393</td>
      <td>437.755764</td>
      <td>437.755764</td>
      <td>443.401427</td>
      <td>440.056019</td>
      <td>437.755764</td>
    </tr>
  </tbody>
</table>
<p>18205 rows × 254 columns</p>
</div>




```python
# Creating a pickle file for user-user recommendation system
import pickle 
pickle.dump(user_final_rating,open(mydrive_path+'pickle_file/user_final_rating.pkl','wb'))
```


```python
d = user_final_rating
d.loc['piggyboy420'].sort_values(ascending=False)[:20]
```




    name
    0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest                          0.0
    Pleasant Hearth 7.5 Steel Grate, 30 5 Bar - Black                                      0.0
    Olivella Bar Soap - 3.52 Oz                                                            0.0
    Orajel Maximum Strength Toothache Pain Relief Liquid                                   0.0
    Pantene Color Preserve Volume Shampoo, 25.4oz                                          0.0
    Pantene Pro-V Expert Collection Age Defy Conditioner                                   0.0
    Pearhead Id Bracelet Frame                                                             0.0
    Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack        0.0
    Physicians Formula Mineral Wear Talc-Free Mineral Correcting Powder, Creamy Natural    0.0
    Physicians Formula Powder Palette Mineral Glow Pearls, Translucent Pearl               0.0
    Pinaud Clubman Styling Gel, Superhold                                                  0.0
    Pink Friday: Roman Reloaded Re-Up (w/dvd)                                              0.0
    Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)                    0.0
    Plano Mini-Magnum 13-Compartment Tackle Box                                            0.0
    Pleasant Hearth 1,800 sq ft Wood Burning Stove with Blower, Medium, LWS-127201         0.0
    Pleasant Hearth Diamond Fireplace Screen - Espresso                                    0.0
    Olay Moisturizing Lotion For Sensitive Skin                                            0.0
    Plum Organics Just Prunes                                                              0.0
    Pocket Watch Wall Clock Distressed Black - Yosemite Home Decor174                      0.0
    Post Bound Jumbo Album - Burgundy (11x14)                                              0.0
    Name: piggyboy420, dtype: float64



### Evaluation for user-user recommendation system


```python
## Evaluation
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape
```




    (2006, 15)




```python
corr_df = pd.DataFrame(user_corr_matrix)
```


```python
corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>18195</th>
      <th>18196</th>
      <th>18197</th>
      <th>18198</th>
      <th>18199</th>
      <th>18200</th>
      <th>18201</th>
      <th>18202</th>
      <th>18203</th>
      <th>18204</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18200</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18201</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18202</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18203</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18204</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>18205 rows × 18205 columns</p>
</div>




```python
corr_df['user_name'] = normalized_df.index
corr_df.set_index('user_name',inplace=True)
corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>18195</th>
      <th>18196</th>
      <th>18197</th>
      <th>18198</th>
      <th>18199</th>
      <th>18200</th>
      <th>18201</th>
      <th>18202</th>
      <th>18203</th>
      <th>18204</th>
    </tr>
    <tr>
      <th>user_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00sab00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>02dakota</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>02deuce</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>0325home</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>06stidriver</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>zuttle</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zwithanx</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zxcsdfd</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zyiah4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zzdiane</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>18205 rows × 18205 columns</p>
</div>




```python
list_name = common.reviews_username.tolist()
```


```python
corr_df.columns = normalized_df.index.tolist()
corr_df.columns
```




    Index(['00sab00', '02dakota', '02deuce', '0325home', '06stidriver', '1.11E+24',
           '1085', '10ten', '11111111aaaaaaaaaaaaaaaaa', '11677j',
           ...
           'zowie', 'zozo0o', 'zsazsa', 'zt313', 'zubb', 'zuttle', 'zwithanx',
           'zxcsdfd', 'zyiah4', 'zzdiane'],
          dtype='object', length=18205)




```python
corr_df1 = corr_df[corr_df.index.isin(list_name)]
corr_df1.shape
```




    (1687, 18205)




```python
corr_df2 = corr_df1.T[corr_df1.T.index.isin(list_name)]
corr_df3 = corr_df2.T
corr_df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1234</th>
      <th>123charlie</th>
      <th>143st</th>
      <th>1943</th>
      <th>4cloroxl</th>
      <th>50cal</th>
      <th>7inthenest</th>
      <th>aac06002</th>
      <th>aaron</th>
      <th>abby</th>
      <th>...</th>
      <th>yeya</th>
      <th>ygtz</th>
      <th>yohnie1</th>
      <th>yshan</th>
      <th>yucky111</th>
      <th>yummy</th>
      <th>yvonne</th>
      <th>zburt5</th>
      <th>zebras</th>
      <th>zippy</th>
    </tr>
    <tr>
      <th>user_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1234</th>
      <td>1.000000</td>
      <td>0.684558</td>
      <td>0.0</td>
      <td>0.223114</td>
      <td>0.0</td>
      <td>0.511718</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.837534</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.278697</td>
    </tr>
    <tr>
      <th>123charlie</th>
      <td>0.684558</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.656818</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>143st</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>0.223114</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.444381</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.242522</td>
    </tr>
    <tr>
      <th>4cloroxl</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1687 columns</p>
</div>




```python
common_user_tb = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',data=common)
```


```python
common_user_tb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>Aussie Aussome Volume Shampoo, 13.5 Oz</th>
      <th>Australian Gold Exotic Blend Lotion, SPF 4</th>
      <th>Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz</th>
      <th>Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter</th>
      <th>Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix</th>
      <th>BRIDGESTONE 130/70ZR18M/C(63W)FRONT EXEDRA G851, CRUISER RADL</th>
      <th>Banana Boat Sunless Summer Color Self Tanning Lotion, Light To Medium</th>
      <th>Barielle Nail Rebuilding Protein</th>
      <th>...</th>
      <th>Vaseline Intensive Care Healthy Hands Stronger Nails</th>
      <th>Vaseline Intensive Care Lip Therapy Cocoa Butter</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1234</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>123charlie</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>143st</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4cloroxl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>yummy</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yvonne</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zburt5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zebras</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zippy</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1687 rows × 120 columns</p>
</div>




```python
corr_df3[corr_df3<0] = 0
common_user_rating =  np.dot(corr_df3,common_user_tb.fillna(0))
common_user_rating
```




    array([[7.4987797 , 1.20864234, 0.        , ..., 7.64092148, 0.92858292,
            0.        ],
           [1.74826633, 0.        , 0.        , ..., 3.36140237, 0.        ,
            0.        ],
           [5.67106405, 5.        , 0.        , ..., 5.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [2.87531862, 0.        , 0.        , ..., 6.10152667, 0.        ,
            0.        ],
           [6.79815093, 1.31072544, 1.07870809, ..., 5.89883029, 1.01174853,
            0.        ]])




```python
dummy_test = common.copy()
dummy_test['reviews_rating'] =dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',data=dummy_test).fillna(0)
dummy_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>Aussie Aussome Volume Shampoo, 13.5 Oz</th>
      <th>Australian Gold Exotic Blend Lotion, SPF 4</th>
      <th>Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz</th>
      <th>Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter</th>
      <th>Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix</th>
      <th>BRIDGESTONE 130/70ZR18M/C(63W)FRONT EXEDRA G851, CRUISER RADL</th>
      <th>Banana Boat Sunless Summer Color Self Tanning Lotion, Light To Medium</th>
      <th>Barielle Nail Rebuilding Protein</th>
      <th>...</th>
      <th>Vaseline Intensive Care Healthy Hands Stronger Nails</th>
      <th>Vaseline Intensive Care Lip Therapy Cocoa Butter</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1234</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>123charlie</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>143st</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4cloroxl</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>yummy</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>yvonne</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zburt5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zebras</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zippy</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1687 rows × 120 columns</p>
</div>




```python
common_user_pred_ratings =  np.multiply(common_user_rating,dummy_test)
common_user_pred_ratings
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>Aussie Aussome Volume Shampoo, 13.5 Oz</th>
      <th>Australian Gold Exotic Blend Lotion, SPF 4</th>
      <th>Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz</th>
      <th>Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter</th>
      <th>Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix</th>
      <th>BRIDGESTONE 130/70ZR18M/C(63W)FRONT EXEDRA G851, CRUISER RADL</th>
      <th>Banana Boat Sunless Summer Color Self Tanning Lotion, Light To Medium</th>
      <th>Barielle Nail Rebuilding Protein</th>
      <th>...</th>
      <th>Vaseline Intensive Care Healthy Hands Stronger Nails</th>
      <th>Vaseline Intensive Care Lip Therapy Cocoa Butter</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1234</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>123charlie</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>143st</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68.109936</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4cloroxl</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>yummy</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>yvonne</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zburt5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zebras</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zippy</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1687 rows × 120 columns</p>
</div>




```python
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_pred_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)
y.shape
```

    MinMaxScaler(feature_range=(1, 5))
    [[nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]
     ...
     [nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]]
    

    /Users/jyoc/Library/Python/3.8/lib/python/site-packages/sklearn/preprocessing/_data.py:461: RuntimeWarning:
    
    All-NaN slice encountered
    
    /Users/jyoc/Library/Python/3.8/lib/python/site-packages/sklearn/preprocessing/_data.py:462: RuntimeWarning:
    
    All-NaN slice encountered
    
    




    (1687, 120)




```python
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))
```


```python
total_non_nan
```




    1787




```python
common_pivot = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',data=common)
common_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>Aussie Aussome Volume Shampoo, 13.5 Oz</th>
      <th>Australian Gold Exotic Blend Lotion, SPF 4</th>
      <th>Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz</th>
      <th>Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter</th>
      <th>Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix</th>
      <th>BRIDGESTONE 130/70ZR18M/C(63W)FRONT EXEDRA G851, CRUISER RADL</th>
      <th>Banana Boat Sunless Summer Color Self Tanning Lotion, Light To Medium</th>
      <th>Barielle Nail Rebuilding Protein</th>
      <th>...</th>
      <th>Vaseline Intensive Care Healthy Hands Stronger Nails</th>
      <th>Vaseline Intensive Care Lip Therapy Cocoa Butter</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1234</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>123charlie</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>143st</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4cloroxl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>yummy</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yvonne</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zburt5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zebras</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zippy</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1687 rows × 120 columns</p>
</div>




```python
rmse = (sum(sum((common_pivot -  y )**2))/total_non_nan)**0.5
print(rmse)
```

    1.8755890184650452
    

## Item and Item recommendation system


```python
train_pivot_ii = train_pivot1.T
train_pivot_ii
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>reviews_username</th>
      <th>00sab00</th>
      <th>02dakota</th>
      <th>02deuce</th>
      <th>0325home</th>
      <th>06stidriver</th>
      <th>1.11E+24</th>
      <th>1085</th>
      <th>10ten</th>
      <th>11111111aaaaaaaaaaaaaaaaa</th>
      <th>11677j</th>
      <th>...</th>
      <th>zowie</th>
      <th>zozo0o</th>
      <th>zsazsa</th>
      <th>zt313</th>
      <th>zubb</th>
      <th>zuttle</th>
      <th>zwithanx</th>
      <th>zxcsdfd</th>
      <th>zyiah4</th>
      <th>zzdiane</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 18205 columns</p>
</div>




```python
item_corr_matrix, normalized_item_df = cosine_similarity(train_pivot_ii)
item_corr_matrix.shape
```




    (254, 254)




```python
item_pred_rating = np.dot((train_pivot_ii.fillna(0)).T,item_corr_matrix)
item_pred_rating[item_pred_rating<0] = 0
item_pred_rating
```




    array([[1.04268435, 0.82942606, 0.90407857, ..., 0.8340146 , 0.81433002,
            0.79418324],
           [1.02447686, 0.82044582, 0.89224956, ..., 0.75605699, 0.74593788,
            0.77182255],
           [1.03032533, 0.82731907, 0.89604918, ..., 0.7795106 , 0.76225108,
            0.78034392],
           ...,
           [1.00512388, 0.6688738 , 0.87967635, ..., 0.59972721, 0.70334103,
            0.70927094],
           [1.00512388, 0.6688738 , 0.87967635, ..., 0.59972721, 0.70334103,
            0.70927094],
           [1.03749796, 0.8109133 , 0.90070908, ..., 0.81815776, 0.81746936,
            0.78245846]])




```python
#final rating for items
item_final_rating = np.multiply(item_pred_rating,train_pivot)
item_final_rating
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>5302050 15/16 FCT/HOSE ADAPTOR</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>...</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Walkers Stem Ginger Shortbread</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>reviews_username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00sab00</th>
      <td>1.042684</td>
      <td>0.829426</td>
      <td>0.904079</td>
      <td>0.865297</td>
      <td>1.257602</td>
      <td>0.864404</td>
      <td>0.932152</td>
      <td>0.904079</td>
      <td>0.904079</td>
      <td>0.776925</td>
      <td>...</td>
      <td>0.629687</td>
      <td>0.904079</td>
      <td>0.948084</td>
      <td>0.932152</td>
      <td>0.987044</td>
      <td>0.786945</td>
      <td>0.874076</td>
      <td>0.834015</td>
      <td>0.814330</td>
      <td>0.794183</td>
    </tr>
    <tr>
      <th>02dakota</th>
      <td>1.024477</td>
      <td>0.820446</td>
      <td>0.892250</td>
      <td>0.848678</td>
      <td>1.146304</td>
      <td>0.847674</td>
      <td>0.923788</td>
      <td>0.892250</td>
      <td>0.892250</td>
      <td>0.739312</td>
      <td>...</td>
      <td>0.566040</td>
      <td>0.892250</td>
      <td>0.999030</td>
      <td>0.923788</td>
      <td>0.998108</td>
      <td>0.760634</td>
      <td>0.858542</td>
      <td>0.756057</td>
      <td>0.745938</td>
      <td>0.771823</td>
    </tr>
    <tr>
      <th>02deuce</th>
      <td>1.030325</td>
      <td>0.827319</td>
      <td>0.896049</td>
      <td>0.854016</td>
      <td>1.148991</td>
      <td>0.853048</td>
      <td>0.926475</td>
      <td>0.896049</td>
      <td>0.896049</td>
      <td>0.748478</td>
      <td>...</td>
      <td>0.586484</td>
      <td>0.896049</td>
      <td>0.987127</td>
      <td>0.926475</td>
      <td>0.988945</td>
      <td>0.769085</td>
      <td>0.863532</td>
      <td>0.779511</td>
      <td>0.762251</td>
      <td>0.780344</td>
    </tr>
    <tr>
      <th>0325home</th>
      <td>1.005124</td>
      <td>0.668874</td>
      <td>0.879676</td>
      <td>0.831013</td>
      <td>1.137414</td>
      <td>0.829892</td>
      <td>0.914898</td>
      <td>0.879676</td>
      <td>0.879676</td>
      <td>0.599313</td>
      <td>...</td>
      <td>0.527652</td>
      <td>0.879676</td>
      <td>0.923927</td>
      <td>0.914898</td>
      <td>0.927441</td>
      <td>0.732668</td>
      <td>0.842030</td>
      <td>0.599727</td>
      <td>0.703341</td>
      <td>0.709271</td>
    </tr>
    <tr>
      <th>06stidriver</th>
      <td>1.005124</td>
      <td>0.668874</td>
      <td>0.879676</td>
      <td>0.831013</td>
      <td>1.137414</td>
      <td>0.829892</td>
      <td>0.914898</td>
      <td>0.879676</td>
      <td>0.879676</td>
      <td>0.599313</td>
      <td>...</td>
      <td>0.527652</td>
      <td>0.879676</td>
      <td>0.923927</td>
      <td>0.914898</td>
      <td>0.927441</td>
      <td>0.732668</td>
      <td>0.842030</td>
      <td>0.599727</td>
      <td>0.703341</td>
      <td>0.709271</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>zuttle</th>
      <td>1.037404</td>
      <td>0.821633</td>
      <td>0.900648</td>
      <td>0.860477</td>
      <td>1.152243</td>
      <td>0.859552</td>
      <td>0.929727</td>
      <td>0.900648</td>
      <td>0.900648</td>
      <td>0.779304</td>
      <td>...</td>
      <td>0.624816</td>
      <td>0.900648</td>
      <td>0.944688</td>
      <td>0.929727</td>
      <td>0.953126</td>
      <td>0.779314</td>
      <td>0.869571</td>
      <td>0.812825</td>
      <td>0.781995</td>
      <td>0.782245</td>
    </tr>
    <tr>
      <th>zwithanx</th>
      <td>1.005124</td>
      <td>0.668874</td>
      <td>0.879676</td>
      <td>0.831013</td>
      <td>1.137414</td>
      <td>0.829892</td>
      <td>0.914898</td>
      <td>0.879676</td>
      <td>0.879676</td>
      <td>0.599313</td>
      <td>...</td>
      <td>0.527652</td>
      <td>0.879676</td>
      <td>0.923927</td>
      <td>0.914898</td>
      <td>0.927441</td>
      <td>0.732668</td>
      <td>0.842030</td>
      <td>0.599727</td>
      <td>0.703341</td>
      <td>0.709271</td>
    </tr>
    <tr>
      <th>zxcsdfd</th>
      <td>1.005124</td>
      <td>0.668874</td>
      <td>0.879676</td>
      <td>0.831013</td>
      <td>1.137414</td>
      <td>0.829892</td>
      <td>0.914898</td>
      <td>0.879676</td>
      <td>0.879676</td>
      <td>0.599313</td>
      <td>...</td>
      <td>0.527652</td>
      <td>0.879676</td>
      <td>0.923927</td>
      <td>0.914898</td>
      <td>0.927441</td>
      <td>0.732668</td>
      <td>0.842030</td>
      <td>0.599727</td>
      <td>0.703341</td>
      <td>0.709271</td>
    </tr>
    <tr>
      <th>zyiah4</th>
      <td>1.005124</td>
      <td>0.668874</td>
      <td>0.879676</td>
      <td>0.831013</td>
      <td>1.137414</td>
      <td>0.829892</td>
      <td>0.914898</td>
      <td>0.879676</td>
      <td>0.879676</td>
      <td>0.599313</td>
      <td>...</td>
      <td>0.527652</td>
      <td>0.879676</td>
      <td>0.923927</td>
      <td>0.914898</td>
      <td>0.927441</td>
      <td>0.732668</td>
      <td>0.842030</td>
      <td>0.599727</td>
      <td>0.703341</td>
      <td>0.709271</td>
    </tr>
    <tr>
      <th>zzdiane</th>
      <td>1.037498</td>
      <td>0.810913</td>
      <td>0.900709</td>
      <td>0.860563</td>
      <td>1.358154</td>
      <td>0.859638</td>
      <td>0.929770</td>
      <td>0.900709</td>
      <td>0.900709</td>
      <td>0.777873</td>
      <td>...</td>
      <td>0.611557</td>
      <td>0.900709</td>
      <td>0.944749</td>
      <td>0.929770</td>
      <td>1.012633</td>
      <td>0.779450</td>
      <td>0.869651</td>
      <td>0.818158</td>
      <td>0.817469</td>
      <td>0.782458</td>
    </tr>
  </tbody>
</table>
<p>18205 rows × 254 columns</p>
</div>




```python
d_item = item_final_rating
d_item.loc['piggyboy420'].sort_values(ascending=False)[:20]
```




    name
    Cantu Coconut Milk Shine Hold Mist - 8oz                                                  2.320392
    Newman's Own Organics Licorice Twist, Black 5oz                                           2.306636
    Sea Gull Lighting Six Light Bath Sconce/vanity - Brushed Nickel                           2.264279
    Naturtint Nutrideep Multiplier Protective Cream                                           2.258941
    Smead174 Recycled Letter Size Manila File Backs W/prong Fasteners, 2 Capacity, 100/box    2.182178
    Pink Friday: Roman Reloaded Re-Up (w/dvd)                                                 2.182178
    Chips Deluxe Soft 'n Chewy Cookies                                                        2.072777
    Home Health Hairever Shampoo                                                              1.837045
    The Seaweed Bath Co. Argan Conditioner, Smoothing Citrus                                  1.789204
    Diet Canada Dry Ginger Ale - 12pk/12 Fl Oz Cans                                           1.764644
    Tostitos Bite Size Tortilla Chips                                                         1.743292
    Mill Creek Aloe Vera & Paba Lotion                                                        1.689716
    Various - Country's Greatest Gospel:Gold Ed (cd)                                          1.626383
    Colorganics Lipstick, Cayenne                                                             1.559491
    Newman's Own Balsamic Vinaigrette, 16.0oz                                                 1.556667
    Soothing Touch Lemon Cardamom Vegan Lip Balm .25 Oz                                       1.497470
    Chester's Cheese Flavored Puffcorn Snacks                                                 1.452572
    Chobani174 Strawberry On The Bottom Non-Fat Greek Yogurt - 5.3oz                          1.437817
    Various - Red Hot Blue:Tribute To Cole Porter (cd)                                        1.407393
    Cheetos Crunchy Flamin' Hot Cheese Flavored Snacks                                        1.392905
    Name: piggyboy420, dtype: float64



### Evaluation for item and item


```python
common_item = test[test.name.isin(train.name)]
common_item
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>brand</th>
      <th>categories</th>
      <th>manufacturer</th>
      <th>name</th>
      <th>reviews_date</th>
      <th>reviews_didPurchase</th>
      <th>reviews_doRecommend</th>
      <th>reviews_rating</th>
      <th>reviews_text</th>
      <th>reviews_title</th>
      <th>reviews_userCity</th>
      <th>reviews_userProvince</th>
      <th>reviews_username</th>
      <th>user_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19154</th>
      <td>AVpfJP1C1cnluZ0-e3Xy</td>
      <td>Clorox</td>
      <td>Household Chemicals,Household Cleaners,Bath &amp; Shower Cleaner,Household Essentials,Cleaning Supplies,Bathroom Cleaners,Prime Pantry,Bathroom,Featured Brands,Home And Storage &amp; Org,Clorox,All-purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Target Restock,Food &amp; Grocery</td>
      <td>AmazonUs/CLOO7</td>
      <td>Clorox Disinfecting Bathroom Cleaner</td>
      <td>2014-12-30T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>Very powerful, great at removing stains, and so convenient to use...just grab and go! This review was collected as part of a promotion.</td>
      <td>Clorox Rocks</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>briley</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>22871</th>
      <td>AVpfov9TLJeJML43A7B0</td>
      <td>Bisquick</td>
      <td>Food &amp; Beverage,Baking &amp; Cooking Essentials,Baking Essentials,Baking Mixes,Breakfast &amp; Cereal,Pancakes, Waffles &amp; Baking Mixes,Food,Pancake &amp; Waffle Mix,Grocery &amp; Gourmet Food,Cooking &amp; Baking,Biscuits,Featured Brands,Grocery,General Mills,Food &amp; Grocery,Breakfast Foods,Pancake Mixes &amp; Syrup,More Dry Mixes,Baking</td>
      <td>GENERAL MILLS SALES, INC.</td>
      <td>Bisquick Original Pancake And Baking Mix - 40oz</td>
      <td>2012-07-25T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>you can do sooooo much with this product..biscuits, impossible pies, shortcake, deserts, think them up yourself or go to the Betty Crocker recipe site to get some ideas... I looked at a few and thought of many more on my own...</td>
      <td>Alot of a good thing!!!</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foxfire61</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>11830</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2014-12-05T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>I love the lemon fresh smell it leaves. I know that what ever I use the Clorox wipe it will be clean. It is extremely important since my son is in the process of getting chemotherapy. Must have a germ free home. This review was collected as part of a promotion.</td>
      <td>Keep Cancer Home Germ Free</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>margies</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>707</th>
      <td>AV1YGDqsGV-KLJ3adc-O</td>
      <td>Windex</td>
      <td>Household Essentials,Cleaning Supplies,Glass Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Featured Brands,Home And Storage &amp; Org,Thanksgathering,All-purpose Cleaners,Target Restock,Food &amp; Grocery,Glass &amp; Window</td>
      <td>Windex</td>
      <td>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</td>
      <td>2015-08-18T00:00:00.000Z</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>Windex used to be the best but whatever they have done to the formula has now made it horrible. It leaves a film on all the windows, mirrors, and glass table tops. What happened Waste of my money. Will not buy and do not recommend.</td>
      <td>no longer a good glass cleaner</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mel</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>20513</th>
      <td>AVpfJP1C1cnluZ0-e3Xy</td>
      <td>Clorox</td>
      <td>Household Chemicals,Household Cleaners,Bath &amp; Shower Cleaner,Household Essentials,Cleaning Supplies,Bathroom Cleaners,Prime Pantry,Bathroom,Featured Brands,Home And Storage &amp; Org,Clorox,All-purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Target Restock,Food &amp; Grocery</td>
      <td>AmazonUs/CLOO7</td>
      <td>Clorox Disinfecting Bathroom Cleaner</td>
      <td>2014-04-07T12:05:20.000Z</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>I have two sons and they make a mess in their bathroom and this product handles it with no problem</td>
      <td>Great In Showers</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jillybeansoccermom</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28158</th>
      <td>AVpfRTh1ilAPnD_xYic2</td>
      <td>Disney</td>
      <td>Movies, Music &amp; Books,Movies,Kids' &amp; Family,Ways To Shop Entertainment,Movies &amp; Tv On Blu-Ray,Movies &amp; TV,Disney,Blu-ray,Children &amp; Family,Movies &amp; Music,Movies &amp; TV Shows,Electronics, Tech Toys, Movies, Music,Blu-Rays,See ALL Blu-Ray,Frys</td>
      <td>Walt Disney</td>
      <td>Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)</td>
      <td>2014-11-07T00:00:00.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>Great family movie. My kids loved it. Goes good with the other cars type.</td>
      <td>great movie</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tony</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7350</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2015-01-28T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>4</td>
      <td>Good product and very convenient to have around the house. This review was collected as part of a promotion.</td>
      <td>convenient</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>heggemister</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>16974</th>
      <td>AVpf9pzn1cnluZ0-uNTM</td>
      <td>Lundberg</td>
      <td>Food,Packaged Foods,Packaged Grains,Rice,Brown Rice,Meal Solutions, Grains &amp; Pasta,Grains &amp; Rice,Grocery &amp; Gourmet Food,Dried Beans, Grains &amp; Rice,Brown</td>
      <td>Lundberg Family Farms</td>
      <td>Lundberg Wehani Rice, 25lb</td>
      <td>2015-09-17T00:00:00.000Z</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>THIS RICE IS THE ONLY ONE I EAT AT HOME. SO NUTRITIOUS!</td>
      <td>Five Stars</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>byindubstylo</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>19418</th>
      <td>AVpfJP1C1cnluZ0-e3Xy</td>
      <td>Clorox</td>
      <td>Household Chemicals,Household Cleaners,Bath &amp; Shower Cleaner,Household Essentials,Cleaning Supplies,Bathroom Cleaners,Prime Pantry,Bathroom,Featured Brands,Home And Storage &amp; Org,Clorox,All-purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Target Restock,Food &amp; Grocery</td>
      <td>AmazonUs/CLOO7</td>
      <td>Clorox Disinfecting Bathroom Cleaner</td>
      <td>2012-01-26T00:00:00.000Z</td>
      <td>False</td>
      <td>True</td>
      <td>5</td>
      <td>i use the clorox wipes for everything. they are great for disinfecting and cleaning up messes. it is a great product.</td>
      <td>great for anything</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nack101</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>8667</th>
      <td>AVpf3VOfilAPnD_xjpun</td>
      <td>Clorox</td>
      <td>Household Essentials,Cleaning Supplies,Kitchen Cleaners,Cleaning Wipes,All-Purpose Cleaners,Health &amp; Household,Household Supplies,Household Cleaning,Ways To Shop,Classroom Essentials,Featured Brands,Home And Storage &amp; Org,Clorox,Glass Cleaners,Surface Care &amp; Protection,Business &amp; Industrial,Cleaning &amp; Janitorial Supplies,Cleaners &amp; Disinfectants,Cleaning Wipes &amp; Pads,Cleaning Solutions,Housewares,Target Restock,Food &amp; Grocery,Paper Goods,Wipes,All Purpose Cleaners</td>
      <td>Clorox</td>
      <td>Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total</td>
      <td>2012-01-26T14:05:59.000Z</td>
      <td>NaN</td>
      <td>True</td>
      <td>5</td>
      <td>Quicks and Easy. Great smell. House smells clean. Like the ease and knowing my house is germ free for my family.</td>
      <td>Quick And Easy!</td>
      <td>Lincoln</td>
      <td>NaN</td>
      <td>lacey</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
<p>8983 rows × 15 columns</p>
</div>




```python
common_item_pivot = common_item.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating').T

common_item_pivot.shape
```




    (206, 8379)




```python
item_corr_df = pd.DataFrame(item_corr_matrix)
item_corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>244</th>
      <th>245</th>
      <th>246</th>
      <th>247</th>
      <th>248</th>
      <th>249</th>
      <th>250</th>
      <th>251</th>
      <th>252</th>
      <th>253</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>-0.001188</td>
      <td>-0.000169</td>
      <td>-0.000238</td>
      <td>-0.000120</td>
      <td>-0.000239</td>
      <td>-0.000120</td>
      <td>-0.000169</td>
      <td>-0.000169</td>
      <td>-0.001148</td>
      <td>...</td>
      <td>-0.000910</td>
      <td>-0.000169</td>
      <td>-0.000167</td>
      <td>-0.000120</td>
      <td>-0.000207</td>
      <td>-0.000376</td>
      <td>-0.000222</td>
      <td>-0.001611</td>
      <td>-0.000726</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001188</td>
      <td>1.000000</td>
      <td>-0.000772</td>
      <td>-0.001084</td>
      <td>-0.000546</td>
      <td>-0.001091</td>
      <td>-0.000546</td>
      <td>-0.000772</td>
      <td>-0.000772</td>
      <td>-0.005241</td>
      <td>...</td>
      <td>-0.004152</td>
      <td>-0.000772</td>
      <td>-0.000764</td>
      <td>-0.000546</td>
      <td>-0.000945</td>
      <td>-0.001716</td>
      <td>-0.001013</td>
      <td>-0.005245</td>
      <td>-0.003313</td>
      <td>-0.002685</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000169</td>
      <td>-0.000772</td>
      <td>1.000000</td>
      <td>-0.000154</td>
      <td>-0.000078</td>
      <td>-0.000155</td>
      <td>-0.000078</td>
      <td>-0.000110</td>
      <td>-0.000110</td>
      <td>-0.000746</td>
      <td>...</td>
      <td>-0.000591</td>
      <td>-0.000110</td>
      <td>-0.000109</td>
      <td>-0.000078</td>
      <td>-0.000135</td>
      <td>-0.000244</td>
      <td>-0.000144</td>
      <td>-0.001047</td>
      <td>-0.000472</td>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.000238</td>
      <td>-0.001084</td>
      <td>-0.000154</td>
      <td>1.000000</td>
      <td>-0.000109</td>
      <td>-0.000218</td>
      <td>-0.000109</td>
      <td>-0.000154</td>
      <td>-0.000154</td>
      <td>-0.001048</td>
      <td>...</td>
      <td>-0.000831</td>
      <td>-0.000154</td>
      <td>-0.000153</td>
      <td>-0.000109</td>
      <td>-0.000189</td>
      <td>-0.000343</td>
      <td>-0.000203</td>
      <td>-0.001471</td>
      <td>-0.000663</td>
      <td>-0.000537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.000120</td>
      <td>-0.000546</td>
      <td>-0.000078</td>
      <td>-0.000109</td>
      <td>1.000000</td>
      <td>-0.000110</td>
      <td>-0.000055</td>
      <td>-0.000078</td>
      <td>-0.000078</td>
      <td>-0.000528</td>
      <td>...</td>
      <td>-0.000418</td>
      <td>-0.000078</td>
      <td>-0.000077</td>
      <td>-0.000055</td>
      <td>-0.000095</td>
      <td>-0.000173</td>
      <td>-0.000102</td>
      <td>-0.000740</td>
      <td>0.170716</td>
      <td>-0.000270</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>249</th>
      <td>-0.000376</td>
      <td>-0.001716</td>
      <td>-0.000244</td>
      <td>-0.000343</td>
      <td>-0.000173</td>
      <td>-0.000346</td>
      <td>-0.000173</td>
      <td>-0.000244</td>
      <td>-0.000244</td>
      <td>-0.001660</td>
      <td>...</td>
      <td>-0.001315</td>
      <td>-0.000244</td>
      <td>-0.000242</td>
      <td>-0.000173</td>
      <td>-0.000299</td>
      <td>1.000000</td>
      <td>-0.000321</td>
      <td>-0.002328</td>
      <td>-0.001049</td>
      <td>-0.000850</td>
    </tr>
    <tr>
      <th>250</th>
      <td>-0.000222</td>
      <td>-0.001013</td>
      <td>-0.000144</td>
      <td>-0.000203</td>
      <td>-0.000102</td>
      <td>-0.000204</td>
      <td>-0.000102</td>
      <td>-0.000144</td>
      <td>-0.000144</td>
      <td>-0.000980</td>
      <td>...</td>
      <td>-0.000776</td>
      <td>-0.000144</td>
      <td>-0.000143</td>
      <td>-0.000102</td>
      <td>-0.000177</td>
      <td>-0.000321</td>
      <td>1.000000</td>
      <td>-0.001375</td>
      <td>-0.000619</td>
      <td>-0.000502</td>
    </tr>
    <tr>
      <th>251</th>
      <td>-0.001611</td>
      <td>-0.005245</td>
      <td>-0.001047</td>
      <td>-0.001471</td>
      <td>-0.000740</td>
      <td>-0.001480</td>
      <td>-0.000740</td>
      <td>-0.001047</td>
      <td>-0.001047</td>
      <td>0.002321</td>
      <td>...</td>
      <td>-0.005632</td>
      <td>-0.001047</td>
      <td>-0.001036</td>
      <td>-0.000740</td>
      <td>-0.001282</td>
      <td>-0.002328</td>
      <td>-0.001375</td>
      <td>1.000000</td>
      <td>-0.004494</td>
      <td>-0.003642</td>
    </tr>
    <tr>
      <th>252</th>
      <td>-0.000726</td>
      <td>-0.003313</td>
      <td>-0.000472</td>
      <td>-0.000663</td>
      <td>0.170716</td>
      <td>-0.000667</td>
      <td>-0.000334</td>
      <td>-0.000472</td>
      <td>-0.000472</td>
      <td>-0.003203</td>
      <td>...</td>
      <td>-0.002538</td>
      <td>-0.000472</td>
      <td>-0.000467</td>
      <td>-0.000334</td>
      <td>-0.000578</td>
      <td>-0.001049</td>
      <td>-0.000619</td>
      <td>-0.004494</td>
      <td>1.000000</td>
      <td>-0.001641</td>
    </tr>
    <tr>
      <th>253</th>
      <td>-0.000588</td>
      <td>-0.002685</td>
      <td>-0.000382</td>
      <td>-0.000537</td>
      <td>-0.000270</td>
      <td>-0.000541</td>
      <td>-0.000270</td>
      <td>-0.000382</td>
      <td>-0.000382</td>
      <td>-0.002596</td>
      <td>...</td>
      <td>-0.002057</td>
      <td>-0.000382</td>
      <td>-0.000378</td>
      <td>-0.000270</td>
      <td>-0.000468</td>
      <td>-0.000850</td>
      <td>-0.000502</td>
      <td>-0.003642</td>
      <td>-0.001641</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 254 columns</p>
</div>




```python
item_corr_df['name'] = normalized_item_df.index
item_corr_df.set_index('name',inplace=True)
item_corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>244</th>
      <th>245</th>
      <th>246</th>
      <th>247</th>
      <th>248</th>
      <th>249</th>
      <th>250</th>
      <th>251</th>
      <th>252</th>
      <th>253</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>1.000000</td>
      <td>-0.001188</td>
      <td>-0.000169</td>
      <td>-0.000238</td>
      <td>-0.000120</td>
      <td>-0.000239</td>
      <td>-0.000120</td>
      <td>-0.000169</td>
      <td>-0.000169</td>
      <td>-0.001148</td>
      <td>...</td>
      <td>-0.000910</td>
      <td>-0.000169</td>
      <td>-0.000167</td>
      <td>-0.000120</td>
      <td>-0.000207</td>
      <td>-0.000376</td>
      <td>-0.000222</td>
      <td>-0.001611</td>
      <td>-0.000726</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>-0.001188</td>
      <td>1.000000</td>
      <td>-0.000772</td>
      <td>-0.001084</td>
      <td>-0.000546</td>
      <td>-0.001091</td>
      <td>-0.000546</td>
      <td>-0.000772</td>
      <td>-0.000772</td>
      <td>-0.005241</td>
      <td>...</td>
      <td>-0.004152</td>
      <td>-0.000772</td>
      <td>-0.000764</td>
      <td>-0.000546</td>
      <td>-0.000945</td>
      <td>-0.001716</td>
      <td>-0.001013</td>
      <td>-0.005245</td>
      <td>-0.003313</td>
      <td>-0.002685</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>-0.000169</td>
      <td>-0.000772</td>
      <td>1.000000</td>
      <td>-0.000154</td>
      <td>-0.000078</td>
      <td>-0.000155</td>
      <td>-0.000078</td>
      <td>-0.000110</td>
      <td>-0.000110</td>
      <td>-0.000746</td>
      <td>...</td>
      <td>-0.000591</td>
      <td>-0.000110</td>
      <td>-0.000109</td>
      <td>-0.000078</td>
      <td>-0.000135</td>
      <td>-0.000244</td>
      <td>-0.000144</td>
      <td>-0.001047</td>
      <td>-0.000472</td>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>-0.000238</td>
      <td>-0.001084</td>
      <td>-0.000154</td>
      <td>1.000000</td>
      <td>-0.000109</td>
      <td>-0.000218</td>
      <td>-0.000109</td>
      <td>-0.000154</td>
      <td>-0.000154</td>
      <td>-0.001048</td>
      <td>...</td>
      <td>-0.000831</td>
      <td>-0.000154</td>
      <td>-0.000153</td>
      <td>-0.000109</td>
      <td>-0.000189</td>
      <td>-0.000343</td>
      <td>-0.000203</td>
      <td>-0.001471</td>
      <td>-0.000663</td>
      <td>-0.000537</td>
    </tr>
    <tr>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <td>-0.000120</td>
      <td>-0.000546</td>
      <td>-0.000078</td>
      <td>-0.000109</td>
      <td>1.000000</td>
      <td>-0.000110</td>
      <td>-0.000055</td>
      <td>-0.000078</td>
      <td>-0.000078</td>
      <td>-0.000528</td>
      <td>...</td>
      <td>-0.000418</td>
      <td>-0.000078</td>
      <td>-0.000077</td>
      <td>-0.000055</td>
      <td>-0.000095</td>
      <td>-0.000173</td>
      <td>-0.000102</td>
      <td>-0.000740</td>
      <td>0.170716</td>
      <td>-0.000270</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>-0.000376</td>
      <td>-0.001716</td>
      <td>-0.000244</td>
      <td>-0.000343</td>
      <td>-0.000173</td>
      <td>-0.000346</td>
      <td>-0.000173</td>
      <td>-0.000244</td>
      <td>-0.000244</td>
      <td>-0.001660</td>
      <td>...</td>
      <td>-0.001315</td>
      <td>-0.000244</td>
      <td>-0.000242</td>
      <td>-0.000173</td>
      <td>-0.000299</td>
      <td>1.000000</td>
      <td>-0.000321</td>
      <td>-0.002328</td>
      <td>-0.001049</td>
      <td>-0.000850</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>-0.000222</td>
      <td>-0.001013</td>
      <td>-0.000144</td>
      <td>-0.000203</td>
      <td>-0.000102</td>
      <td>-0.000204</td>
      <td>-0.000102</td>
      <td>-0.000144</td>
      <td>-0.000144</td>
      <td>-0.000980</td>
      <td>...</td>
      <td>-0.000776</td>
      <td>-0.000144</td>
      <td>-0.000143</td>
      <td>-0.000102</td>
      <td>-0.000177</td>
      <td>-0.000321</td>
      <td>1.000000</td>
      <td>-0.001375</td>
      <td>-0.000619</td>
      <td>-0.000502</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>-0.001611</td>
      <td>-0.005245</td>
      <td>-0.001047</td>
      <td>-0.001471</td>
      <td>-0.000740</td>
      <td>-0.001480</td>
      <td>-0.000740</td>
      <td>-0.001047</td>
      <td>-0.001047</td>
      <td>0.002321</td>
      <td>...</td>
      <td>-0.005632</td>
      <td>-0.001047</td>
      <td>-0.001036</td>
      <td>-0.000740</td>
      <td>-0.001282</td>
      <td>-0.002328</td>
      <td>-0.001375</td>
      <td>1.000000</td>
      <td>-0.004494</td>
      <td>-0.003642</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>-0.000726</td>
      <td>-0.003313</td>
      <td>-0.000472</td>
      <td>-0.000663</td>
      <td>0.170716</td>
      <td>-0.000667</td>
      <td>-0.000334</td>
      <td>-0.000472</td>
      <td>-0.000472</td>
      <td>-0.003203</td>
      <td>...</td>
      <td>-0.002538</td>
      <td>-0.000472</td>
      <td>-0.000467</td>
      <td>-0.000334</td>
      <td>-0.000578</td>
      <td>-0.001049</td>
      <td>-0.000619</td>
      <td>-0.004494</td>
      <td>1.000000</td>
      <td>-0.001641</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>-0.000588</td>
      <td>-0.002685</td>
      <td>-0.000382</td>
      <td>-0.000537</td>
      <td>-0.000270</td>
      <td>-0.000541</td>
      <td>-0.000270</td>
      <td>-0.000382</td>
      <td>-0.000382</td>
      <td>-0.002596</td>
      <td>...</td>
      <td>-0.002057</td>
      <td>-0.000382</td>
      <td>-0.000378</td>
      <td>-0.000270</td>
      <td>-0.000468</td>
      <td>-0.000850</td>
      <td>-0.000502</td>
      <td>-0.003642</td>
      <td>-0.001641</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 254 columns</p>
</div>




```python
list_items = common_item.name.tolist()
item_corr_df.columns = normalized_item_df.index.tolist()
item_corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>5302050 15/16 FCT/HOSE ADAPTOR</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>...</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Walkers Stem Ginger Shortbread</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>1.000000</td>
      <td>-0.001188</td>
      <td>-0.000169</td>
      <td>-0.000238</td>
      <td>-0.000120</td>
      <td>-0.000239</td>
      <td>-0.000120</td>
      <td>-0.000169</td>
      <td>-0.000169</td>
      <td>-0.001148</td>
      <td>...</td>
      <td>-0.000910</td>
      <td>-0.000169</td>
      <td>-0.000167</td>
      <td>-0.000120</td>
      <td>-0.000207</td>
      <td>-0.000376</td>
      <td>-0.000222</td>
      <td>-0.001611</td>
      <td>-0.000726</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>-0.001188</td>
      <td>1.000000</td>
      <td>-0.000772</td>
      <td>-0.001084</td>
      <td>-0.000546</td>
      <td>-0.001091</td>
      <td>-0.000546</td>
      <td>-0.000772</td>
      <td>-0.000772</td>
      <td>-0.005241</td>
      <td>...</td>
      <td>-0.004152</td>
      <td>-0.000772</td>
      <td>-0.000764</td>
      <td>-0.000546</td>
      <td>-0.000945</td>
      <td>-0.001716</td>
      <td>-0.001013</td>
      <td>-0.005245</td>
      <td>-0.003313</td>
      <td>-0.002685</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>-0.000169</td>
      <td>-0.000772</td>
      <td>1.000000</td>
      <td>-0.000154</td>
      <td>-0.000078</td>
      <td>-0.000155</td>
      <td>-0.000078</td>
      <td>-0.000110</td>
      <td>-0.000110</td>
      <td>-0.000746</td>
      <td>...</td>
      <td>-0.000591</td>
      <td>-0.000110</td>
      <td>-0.000109</td>
      <td>-0.000078</td>
      <td>-0.000135</td>
      <td>-0.000244</td>
      <td>-0.000144</td>
      <td>-0.001047</td>
      <td>-0.000472</td>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>-0.000238</td>
      <td>-0.001084</td>
      <td>-0.000154</td>
      <td>1.000000</td>
      <td>-0.000109</td>
      <td>-0.000218</td>
      <td>-0.000109</td>
      <td>-0.000154</td>
      <td>-0.000154</td>
      <td>-0.001048</td>
      <td>...</td>
      <td>-0.000831</td>
      <td>-0.000154</td>
      <td>-0.000153</td>
      <td>-0.000109</td>
      <td>-0.000189</td>
      <td>-0.000343</td>
      <td>-0.000203</td>
      <td>-0.001471</td>
      <td>-0.000663</td>
      <td>-0.000537</td>
    </tr>
    <tr>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <td>-0.000120</td>
      <td>-0.000546</td>
      <td>-0.000078</td>
      <td>-0.000109</td>
      <td>1.000000</td>
      <td>-0.000110</td>
      <td>-0.000055</td>
      <td>-0.000078</td>
      <td>-0.000078</td>
      <td>-0.000528</td>
      <td>...</td>
      <td>-0.000418</td>
      <td>-0.000078</td>
      <td>-0.000077</td>
      <td>-0.000055</td>
      <td>-0.000095</td>
      <td>-0.000173</td>
      <td>-0.000102</td>
      <td>-0.000740</td>
      <td>0.170716</td>
      <td>-0.000270</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>-0.000376</td>
      <td>-0.001716</td>
      <td>-0.000244</td>
      <td>-0.000343</td>
      <td>-0.000173</td>
      <td>-0.000346</td>
      <td>-0.000173</td>
      <td>-0.000244</td>
      <td>-0.000244</td>
      <td>-0.001660</td>
      <td>...</td>
      <td>-0.001315</td>
      <td>-0.000244</td>
      <td>-0.000242</td>
      <td>-0.000173</td>
      <td>-0.000299</td>
      <td>1.000000</td>
      <td>-0.000321</td>
      <td>-0.002328</td>
      <td>-0.001049</td>
      <td>-0.000850</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>-0.000222</td>
      <td>-0.001013</td>
      <td>-0.000144</td>
      <td>-0.000203</td>
      <td>-0.000102</td>
      <td>-0.000204</td>
      <td>-0.000102</td>
      <td>-0.000144</td>
      <td>-0.000144</td>
      <td>-0.000980</td>
      <td>...</td>
      <td>-0.000776</td>
      <td>-0.000144</td>
      <td>-0.000143</td>
      <td>-0.000102</td>
      <td>-0.000177</td>
      <td>-0.000321</td>
      <td>1.000000</td>
      <td>-0.001375</td>
      <td>-0.000619</td>
      <td>-0.000502</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>-0.001611</td>
      <td>-0.005245</td>
      <td>-0.001047</td>
      <td>-0.001471</td>
      <td>-0.000740</td>
      <td>-0.001480</td>
      <td>-0.000740</td>
      <td>-0.001047</td>
      <td>-0.001047</td>
      <td>0.002321</td>
      <td>...</td>
      <td>-0.005632</td>
      <td>-0.001047</td>
      <td>-0.001036</td>
      <td>-0.000740</td>
      <td>-0.001282</td>
      <td>-0.002328</td>
      <td>-0.001375</td>
      <td>1.000000</td>
      <td>-0.004494</td>
      <td>-0.003642</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>-0.000726</td>
      <td>-0.003313</td>
      <td>-0.000472</td>
      <td>-0.000663</td>
      <td>0.170716</td>
      <td>-0.000667</td>
      <td>-0.000334</td>
      <td>-0.000472</td>
      <td>-0.000472</td>
      <td>-0.003203</td>
      <td>...</td>
      <td>-0.002538</td>
      <td>-0.000472</td>
      <td>-0.000467</td>
      <td>-0.000334</td>
      <td>-0.000578</td>
      <td>-0.001049</td>
      <td>-0.000619</td>
      <td>-0.004494</td>
      <td>1.000000</td>
      <td>-0.001641</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>-0.000588</td>
      <td>-0.002685</td>
      <td>-0.000382</td>
      <td>-0.000537</td>
      <td>-0.000270</td>
      <td>-0.000541</td>
      <td>-0.000270</td>
      <td>-0.000382</td>
      <td>-0.000382</td>
      <td>-0.002596</td>
      <td>...</td>
      <td>-0.002057</td>
      <td>-0.000382</td>
      <td>-0.000378</td>
      <td>-0.000270</td>
      <td>-0.000468</td>
      <td>-0.000850</td>
      <td>-0.000502</td>
      <td>-0.003642</td>
      <td>-0.001641</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 254 columns</p>
</div>




```python
list_items
```




    ['Clorox Disinfecting Bathroom Cleaner',
     'Bisquick Original Pancake And Baking Mix - 40oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     "Jason Aldean - They Don't Know",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "There's Something About Mary (dvd)",
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Red (special Edition) (dvdvideo)',
     "Burt's Bees Lip Shimmer, Raisin",
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Bathroom Cleaner',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     "Burt's Bees Lip Shimmer, Raisin",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     "Africa's Best No-Lye Dual Conditioning Relaxer System Super",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Red (special Edition) (dvdvideo)',
     "Burt's Bees Lip Shimmer, Raisin",
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Sopranos:Season 6 Part 1 (blu-Ray)',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "L'oreal Paris Advanced Hairstyle Boost It High Lift Creation Spray",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'Kind Dark Chocolate Chunk Gluten Free Granola Bars - 5 Count',
     'Leslie Sansone:Belly Blasting Walk (dvd)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Bathroom Cleaner',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hormel Chili, No Beans',
     "Stacy's Simply Naked Bagel Chips",
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Eagle Fat Free Sweetened Condensed Milk',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Jergens Extra Moisturizing Liquid Hand Wash, 7.5oz',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Coty Airspun Face Powder, Translucent Extra Coverage',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Sabre 2 Pack Door And Window Alarm',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Tostitos Bite Size Tortilla Chips',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     "Burt's Bees Lip Shimmer, Raisin",
     'Just For Men Touch Of Gray Gray Hair Treatment, Black T-55',
     'Pleasant Hearth 1,800 sq ft Wood Burning Stove with Blower, Medium, LWS-127201',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Red (special Edition) (dvdvideo)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     "Burt's Bees Lip Shimmer, Raisin",
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Cococare 100% Natural Castor Oil',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Head & Shoulders Classic Clean Conditioner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Kraus FVS-1007 Single Hole Vessel Bathroom Faucet from the Ramus Collection',
     'Eagle Fat Free Sweetened Condensed Milk',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'Just For Men Touch Of Gray Gray Hair Treatment, Black T-55',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     "L'oreal Paris Advanced Hairstyle TXT IT Hyper-Fix Putty",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Vaseline Intensive Care Lip Therapy Cocoa Butter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Suave Professionals Hair Conditioner, Sleek',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     "Burt's Bees Lip Shimmer, Raisin",
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Chips Ahoy! Original Chocolate Chip - Cookies - Family Size 18.2oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Vicks Vaporub, Regular, 3.53oz',
     "Burt's Bees Lip Shimmer, Raisin",
     'Various Artists - Choo Choo Soul (cd)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Alex Cross (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Chips Ahoy! Original Chocolate Chip - Cookies - Family Size 18.2oz',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Jolly Time Select Premium Yellow Pop Corn',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Banana Boat Sunless Summer Color Self Tanning Lotion, Light To Medium',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Ragu Traditional Pasta Sauce',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     "Jason Aldean - They Don't Know",
     'Sopranos:Season 6 Part 1 (blu-Ray)',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Lundberg Wehani Rice, 25lb',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     "There's Something About Mary (dvd)",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     '100:Complete First Season (blu-Ray)',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Vaseline Intensive Care Lip Therapy Cocoa Butter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Alex Cross (dvdvideo)',
     "Cheetos Crunchy Flamin' Hot Cheese Flavored Snacks",
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Lay's Salt & Vinegar Flavored Potato Chips",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Lundberg Wehani Rice, 25lb',
     'Pleasant Hearth 7.5 Steel Grate, 30 5 Bar - Black',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Suave Professionals Hair Conditioner, Sleek',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Tresemme Kertatin Smooth Infusing Conditioning',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'The Honest Company Laundry Detergent',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Arrid Extra Dry Anti-Perspirant Deodorant Spray Regular',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Tostitos Bite Size Tortilla Chips',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     '100:Complete First Season (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     "Annie's Homegrown Deluxe Elbows & Four Cheese Sauce",
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Kraus FVS-1007 Single Hole Vessel Bathroom Faucet from the Ramus Collection',
     "Johnson's Baby Bubble Bath and Wash, 15oz",
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Red (special Edition) (dvdvideo)',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Chester's Cheese Flavored Puffcorn Snacks",
     'Lysol Concentrate Deodorizing Cleaner, Original Scent',
     'Lundberg Wehani Rice, 25lb',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Caress Moisturizing Body Bar Natural Silk, 4.75oz',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'Ragu Traditional Pasta Sauce',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Bathroom Cleaner',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Caress Moisturizing Body Bar Natural Silk, 4.75oz',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Super Poligrip Denture Adhesive Cream, Ultra Fresh - 2.4 Oz',
     'Caress Moisturizing Body Bar Natural Silk, 4.75oz',
     "Jason Aldean - They Don't Know",
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'BRIDGESTONE 130/70ZR18M/C(63W)FRONT EXEDRA G851, CRUISER RADL',
     'Kendall Comforter And Sheet Set (twin) Aqua - 7pc',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     "Burt's Bees Lip Shimmer, Raisin",
     'Red (special Edition) (dvdvideo)',
     'Just For Men Touch Of Gray Gray Hair Treatment, Black T-55',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Bathroom Cleaner',
     "L'oreal Paris Advanced Hairstyle TXT IT Hyper-Fix Putty",
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Tostitos Bite Size Tortilla Chips',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Lundberg Wehani Rice, 25lb',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Head & Shoulders Classic Clean Conditioner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Lundberg Wehani Rice, 25lb',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Lundberg Wehani Rice, 25lb',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hormel Chili, No Beans',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Lundberg Wehani Rice, 25lb',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Hormel Chili, No Beans',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Vaseline Intensive Care Lip Therapy Cocoa Butter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Care Free Curl Gold Instant Activator',
     'Alex Cross (dvdvideo)',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Clorox Disinfecting Bathroom Cleaner',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Alex Cross (dvdvideo)',
     '100:Complete First Season (blu-Ray)',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     "Burt's Bees Lip Shimmer, Raisin",
     'Vicks Vaporub, Regular, 3.53oz',
     'Just For Men Touch Of Gray Gray Hair Treatment, Black T-55',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "There's Something About Mary (dvd)",
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Vicks Vaporub, Regular, 3.53oz',
     'Clorox Disinfecting Bathroom Cleaner',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Aussie Aussome Volume Shampoo, 13.5 Oz',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Bathroom Cleaner',
     'Ogx Conditioner, Hydrating Teatree Mint',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Orajel Maximum Strength Toothache Pain Relief Liquid',
     'Red (special Edition) (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Equals (blu-Ray)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Red (special Edition) (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Pantene Color Preserve Volume Shampoo, 25.4oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Honest Company Laundry Detergent',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Jason Aldean - They Don't Know",
     'Clorox Disinfecting Bathroom Cleaner',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Hormel Chili, No Beans',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     "Meguiar's Ultimate Quik Detailer 22-Oz.",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Hormel Chili, No Beans',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Alex Cross (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Bathroom Cleaner',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Iman Second To None Stick Foundation, Clay 1',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Red (special Edition) (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Starbucks Iced Expresso Classics Vanilla Latte Coffee Beverage - 40oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     "Burt's Bees Lip Shimmer, Raisin",
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Red (special Edition) (dvdvideo)',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Australian Gold Exotic Blend Lotion, SPF 4',
     'Boraam Sonoma Kitchen Cart With Wire Brush Gray - Maaya Home',
     'Bisquick Original Pancake And Baking Mix - 40oz',
     'Bisquick Original Pancake And Baking Mix - 40oz',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     "Burt's Bees Lip Shimmer, Raisin",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Lundberg Wehani Rice, 25lb',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Yes To Carrots Nourishing Body Wash',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Tresemme Kertatin Smooth Infusing Conditioning',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Chobani174 Strawberry On The Bottom Non-Fat Greek Yogurt - 5.3oz',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Red (special Edition) (dvdvideo)',
     'Diet Canada Dry Ginger Ale - 12pk/12 Fl Oz Cans',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "Burt's Bees Lip Shimmer, Raisin",
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Head & Shoulders Classic Clean Conditioner',
     'Equals (blu-Ray)',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Kind Dark Chocolate Chunk Gluten Free Granola Bars - 5 Count',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Coty Airspun Face Powder, Translucent Extra Coverage',
     "Jason Aldean - They Don't Know",
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Tostitos Bite Size Tortilla Chips',
     'All,bran Complete Wheat Flakes, 18 Oz.',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Chips Ahoy! Original Chocolate Chip - Cookies - Family Size 18.2oz',
     'Lysol Concentrate Deodorizing Cleaner, Original Scent',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Clorox Disinfecting Bathroom Cleaner',
     '100:Complete First Season (blu-Ray)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "Jason Aldean - They Don't Know",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Diet Canada Dry Ginger Ale - 12pk/12 Fl Oz Cans',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Hoover174 Platinum Collection153 Lightweight Bagged Upright Vacuum With Canister - Uh30010com',
     'Red (special Edition) (dvdvideo)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     "Burt's Bees Lip Shimmer, Raisin",
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Lundberg Wehani Rice, 25lb',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'K-Y Love Sensuality Pleasure Gel',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Kind Nut Delight Bar',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix',
     'Lundberg Wehani Rice, 25lb',
     'Chex Muddy Buddies Brownie Supreme Snack Mix',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Care Free Curl Gold Instant Activator',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'Clorox Disinfecting Bathroom Cleaner',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Sizzix Framelits Dies 1by Tim Holtz Bird Crazy-Silver Asst Sizes',
     'Vaseline Intensive Care Healthy Hands Stronger Nails',
     'Bisquick Original Pancake And Baking Mix - 40oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Clorox Disinfecting Bathroom Cleaner',
     'Kraus FVS-1007 Single Hole Vessel Bathroom Faucet from the Ramus Collection',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Creme Of Nature Intensive Conditioning Treatment, 32',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Chips Ahoy! Original Chocolate Chip - Cookies - Family Size 18.2oz',
     'Red (special Edition) (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Lysol Concentrate Deodorizing Cleaner, Original Scent',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Lundberg Wehani Rice, 25lb',
     'Coty Airspun Face Powder, Translucent Extra Coverage',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Red (special Edition) (dvdvideo)',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'Tostitos Bite Size Tortilla Chips',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     "Chester's Cheese Flavored Puffcorn Snacks",
     '100:Complete First Season (blu-Ray)',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     "Burt's Bees Lip Shimmer, Raisin",
     'Bumble Bee Solid White Albacore In Water - 5 Oz',
     'Boraam Sonoma Kitchen Cart With Wire Brush Gray - Maaya Home',
     'Red (special Edition) (dvdvideo)',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     "Jason Aldean - They Don't Know",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Hormel Chili, No Beans',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Tree Hut Shea Body Butters, Coconut Lime, 7 oz',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Bisquick Original Pancake And Baking Mix - 40oz',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Sizzix Framelits Dies 1by Tim Holtz Bird Crazy-Silver Asst Sizes',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Eagle Fat Free Sweetened Condensed Milk',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Alex Cross (dvdvideo)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     "Chester's Cheese Flavored Puffcorn Snacks",
     'Lysol Concentrate Deodorizing Cleaner, Original Scent',
     "Burt's Bees Lip Shimmer, Raisin",
     'Suave Professionals Hair Conditioner, Sleek',
     'Clorox Disinfecting Bathroom Cleaner',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Lundberg Wehani Rice, 25lb',
     'Progresso Traditional Chicken Tuscany Soup',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     "Burt's Bees Lip Shimmer, Raisin",
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Nexxus Exxtra Gel Style Creation Sculptor',
     'Aussie Aussome Volume Shampoo, 13.5 Oz',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Tostitos Bite Size Tortilla Chips',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Bounce Dryer Sheets, Fresh Linen, 160 sheets',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Chips Ahoy! Original Chocolate Chip - Cookies - Family Size 18.2oz',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Olay Regenerist Deep Hydration Regenerating Cream',
     'Storkcraft Tuscany Glider and Ottoman, Beige Cushions, Espresso Finish',
     'Alex Cross (dvdvideo)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Pantene Pro-V Expert Collection Age Defy Conditioner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'oreal Paris Advanced Hairstyle Boost It High Lift Creation Spray",
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Wagan Smartac 80watt Inverter With Usb',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Vaseline Intensive Care Lip Therapy Cocoa Butter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     '100:Complete First Season (blu-Ray)',
     "Johnson's Baby Bubble Bath and Wash, 15oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Jason Aldean - They Don't Know",
     "There's Something About Mary (dvd)",
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Pendaflex174 Divide It Up File Folder, Multi Section, Letter, Assorted, 12/pack',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Mrs. Meyer's174 Lemon Verbena Laundry Scent Booster - 18oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Australian Gold Exotic Blend Lotion, SPF 4',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Dark Shadows (includes Digital Copy) (ultraviolet) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Alex Cross (dvdvideo)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Lundberg Wehani Rice, 25lb',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'My Big Fat Greek Wedding 2 (blu-Ray + Dvd + Digital)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     "Cheetos Crunchy Flamin' Hot Cheese Flavored Snacks",
     'Cuisinart174 Electric Juicer - Stainless Steel Cje-1000',
     'Tostitos Bite Size Tortilla Chips',
     "Cheetos Crunchy Flamin' Hot Cheese Flavored Snacks",
     "Stargate (ws) (ultimate Edition) (director's Cut) (dvdvideo)",
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Axe Dry Anti-Perspirant Deodorant Invisible Solid Phoenix',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Coty Airspun Face Powder, Translucent Extra Coverage',
     'Just For Men Touch Of Gray Gray Hair Treatment, Black T-55',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'Ogx Conditioner, Hydrating Teatree Mint',
     'Lundberg Wehani Rice, 25lb',
     'Aveeno Baby Continuous Protection Lotion Sunscreen with Broad Spectrum SPF 55, 4oz',
     'Ragu Traditional Pasta Sauce',
     'Windex Original Glass Cleaner Refill 67.6oz (2 Liter)',
     "L'oreal Paris Advanced Hairstyle TXT IT Hyper-Fix Putty",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'K-Y Love Sensuality Pleasure Gel',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Hormel Chili, No Beans',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Vicks Vaporub, Regular, 3.53oz',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "There's Something About Mary (dvd)",
     'Ragu Traditional Pasta Sauce',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Bathroom Cleaner',
     'Planes: Fire Rescue (2 Discs) (includes Digital Copy) (blu-Ray/dvd)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Meguiar's Deep Crystal Car Wash 64-Oz.",
     'Clear Scalp & Hair Therapy Total Care Nourishing Shampoo',
     'Shea Moisture Mango & Carrot Kids Extra-Nourishing Conditioner, 8fl Oz',
     'Avery174 Ready Index Contemporary Table Of Contents Divider, 1-8, Multi, Letter',
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Clorox Disinfecting Bathroom Cleaner',
     'Clorox Disinfecting Bathroom Cleaner',
     'Hormel Chili, No Beans',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     "Burt's Bees Lip Shimmer, Raisin",
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Mike Dave Need Wedding Dates (dvd + Digital)',
     'Lundberg Wehani Rice, 25lb',
     "L'or233al Paris Elvive Extraordinary Clay Rebalancing Conditioner - 12.6 Fl Oz",
     'Godzilla 3d Includes Digital Copy Ultraviolet 3d/2d Blu-Ray/dvd',
     'The Resident Evil Collection 5 Discs (blu-Ray)',
     'Lysol Concentrate Deodorizing Cleaner, Original Scent',
     "Burt's Bees Lip Shimmer, Raisin",
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Red (special Edition) (dvdvideo)',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total',
     ...]




```python
item_corr_df1 = item_corr_df[item_corr_df.index.isin(list_items)]
item_corr_df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>42 Dual Drop Leaf Table with 2 Madrid Chairs"</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>5302050 15/16 FCT/HOSE ADAPTOR</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>...</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Walkers Stem Ginger Shortbread</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>1.000000</td>
      <td>-0.001188</td>
      <td>-0.000169</td>
      <td>-0.000238</td>
      <td>-0.000120</td>
      <td>-0.000239</td>
      <td>-0.000120</td>
      <td>-0.000169</td>
      <td>-0.000169</td>
      <td>-0.001148</td>
      <td>...</td>
      <td>-0.000910</td>
      <td>-0.000169</td>
      <td>-0.000167</td>
      <td>-0.000120</td>
      <td>-0.000207</td>
      <td>-0.000376</td>
      <td>-0.000222</td>
      <td>-0.001611</td>
      <td>-0.000726</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>-0.001188</td>
      <td>1.000000</td>
      <td>-0.000772</td>
      <td>-0.001084</td>
      <td>-0.000546</td>
      <td>-0.001091</td>
      <td>-0.000546</td>
      <td>-0.000772</td>
      <td>-0.000772</td>
      <td>-0.005241</td>
      <td>...</td>
      <td>-0.004152</td>
      <td>-0.000772</td>
      <td>-0.000764</td>
      <td>-0.000546</td>
      <td>-0.000945</td>
      <td>-0.001716</td>
      <td>-0.001013</td>
      <td>-0.005245</td>
      <td>-0.003313</td>
      <td>-0.002685</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>-0.000169</td>
      <td>-0.000772</td>
      <td>1.000000</td>
      <td>-0.000154</td>
      <td>-0.000078</td>
      <td>-0.000155</td>
      <td>-0.000078</td>
      <td>-0.000110</td>
      <td>-0.000110</td>
      <td>-0.000746</td>
      <td>...</td>
      <td>-0.000591</td>
      <td>-0.000110</td>
      <td>-0.000109</td>
      <td>-0.000078</td>
      <td>-0.000135</td>
      <td>-0.000244</td>
      <td>-0.000144</td>
      <td>-0.001047</td>
      <td>-0.000472</td>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>-0.000238</td>
      <td>-0.001084</td>
      <td>-0.000154</td>
      <td>1.000000</td>
      <td>-0.000109</td>
      <td>-0.000218</td>
      <td>-0.000109</td>
      <td>-0.000154</td>
      <td>-0.000154</td>
      <td>-0.001048</td>
      <td>...</td>
      <td>-0.000831</td>
      <td>-0.000154</td>
      <td>-0.000153</td>
      <td>-0.000109</td>
      <td>-0.000189</td>
      <td>-0.000343</td>
      <td>-0.000203</td>
      <td>-0.001471</td>
      <td>-0.000663</td>
      <td>-0.000537</td>
    </tr>
    <tr>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <td>-0.000239</td>
      <td>-0.001091</td>
      <td>-0.000155</td>
      <td>-0.000218</td>
      <td>-0.000110</td>
      <td>1.000000</td>
      <td>-0.000110</td>
      <td>-0.000155</td>
      <td>-0.000155</td>
      <td>-0.001055</td>
      <td>...</td>
      <td>-0.000836</td>
      <td>-0.000155</td>
      <td>-0.000154</td>
      <td>-0.000110</td>
      <td>-0.000190</td>
      <td>-0.000346</td>
      <td>-0.000204</td>
      <td>-0.001480</td>
      <td>-0.000667</td>
      <td>-0.000541</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>-0.000376</td>
      <td>-0.001716</td>
      <td>-0.000244</td>
      <td>-0.000343</td>
      <td>-0.000173</td>
      <td>-0.000346</td>
      <td>-0.000173</td>
      <td>-0.000244</td>
      <td>-0.000244</td>
      <td>-0.001660</td>
      <td>...</td>
      <td>-0.001315</td>
      <td>-0.000244</td>
      <td>-0.000242</td>
      <td>-0.000173</td>
      <td>-0.000299</td>
      <td>1.000000</td>
      <td>-0.000321</td>
      <td>-0.002328</td>
      <td>-0.001049</td>
      <td>-0.000850</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>-0.000222</td>
      <td>-0.001013</td>
      <td>-0.000144</td>
      <td>-0.000203</td>
      <td>-0.000102</td>
      <td>-0.000204</td>
      <td>-0.000102</td>
      <td>-0.000144</td>
      <td>-0.000144</td>
      <td>-0.000980</td>
      <td>...</td>
      <td>-0.000776</td>
      <td>-0.000144</td>
      <td>-0.000143</td>
      <td>-0.000102</td>
      <td>-0.000177</td>
      <td>-0.000321</td>
      <td>1.000000</td>
      <td>-0.001375</td>
      <td>-0.000619</td>
      <td>-0.000502</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>-0.001611</td>
      <td>-0.005245</td>
      <td>-0.001047</td>
      <td>-0.001471</td>
      <td>-0.000740</td>
      <td>-0.001480</td>
      <td>-0.000740</td>
      <td>-0.001047</td>
      <td>-0.001047</td>
      <td>0.002321</td>
      <td>...</td>
      <td>-0.005632</td>
      <td>-0.001047</td>
      <td>-0.001036</td>
      <td>-0.000740</td>
      <td>-0.001282</td>
      <td>-0.002328</td>
      <td>-0.001375</td>
      <td>1.000000</td>
      <td>-0.004494</td>
      <td>-0.003642</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>-0.000726</td>
      <td>-0.003313</td>
      <td>-0.000472</td>
      <td>-0.000663</td>
      <td>0.170716</td>
      <td>-0.000667</td>
      <td>-0.000334</td>
      <td>-0.000472</td>
      <td>-0.000472</td>
      <td>-0.003203</td>
      <td>...</td>
      <td>-0.002538</td>
      <td>-0.000472</td>
      <td>-0.000467</td>
      <td>-0.000334</td>
      <td>-0.000578</td>
      <td>-0.001049</td>
      <td>-0.000619</td>
      <td>-0.004494</td>
      <td>1.000000</td>
      <td>-0.001641</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>-0.000588</td>
      <td>-0.002685</td>
      <td>-0.000382</td>
      <td>-0.000537</td>
      <td>-0.000270</td>
      <td>-0.000541</td>
      <td>-0.000270</td>
      <td>-0.000382</td>
      <td>-0.000382</td>
      <td>-0.002596</td>
      <td>...</td>
      <td>-0.002057</td>
      <td>-0.000382</td>
      <td>-0.000378</td>
      <td>-0.000270</td>
      <td>-0.000468</td>
      <td>-0.000850</td>
      <td>-0.000502</td>
      <td>-0.003642</td>
      <td>-0.001641</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>206 rows × 254 columns</p>
</div>




```python
item_corr_df2 = item_corr_df1.T[item_corr_df1.T.index.isin(list_items)]
item_corr_df3 = item_corr_df2.T
item_corr_df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <th>100:Complete First Season (blu-Ray)</th>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <th>Africa's Best No-Lye Dual Conditioning Relaxer System Super</th>
      <th>Alberto VO5 Salon Series Smooth Plus Sleek Shampoo</th>
      <th>Alex Cross (dvdvideo)</th>
      <th>All,bran Complete Wheat Flakes, 18 Oz.</th>
      <th>Ambi Complexion Cleansing Bar</th>
      <th>...</th>
      <th>Vaseline Intensive Care Lip Therapy Cocoa Butter</th>
      <th>Vicks Vaporub, Regular, 3.53oz</th>
      <th>Voortman Sugar Free Fudge Chocolate Chip Cookies</th>
      <th>Wagan Smartac 80watt Inverter With Usb</th>
      <th>Way Basics 3-Shelf Eco Narrow Bookcase Storage Shelf, Espresso - Formaldehyde Free - Lifetime Guarantee</th>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <th>Weleda Everon Lip Balm</th>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>1.000000</td>
      <td>-0.001188</td>
      <td>-0.000169</td>
      <td>-0.000238</td>
      <td>-0.000239</td>
      <td>-0.000169</td>
      <td>-0.000169</td>
      <td>-0.001148</td>
      <td>-0.000337</td>
      <td>-0.000167</td>
      <td>...</td>
      <td>-0.001036</td>
      <td>-0.000910</td>
      <td>-0.000169</td>
      <td>-0.000167</td>
      <td>-0.000207</td>
      <td>-0.000376</td>
      <td>-0.000222</td>
      <td>-0.001611</td>
      <td>-0.000726</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>-0.001188</td>
      <td>1.000000</td>
      <td>-0.000772</td>
      <td>-0.001084</td>
      <td>-0.001091</td>
      <td>-0.000772</td>
      <td>-0.000772</td>
      <td>-0.005241</td>
      <td>-0.001538</td>
      <td>-0.000764</td>
      <td>...</td>
      <td>-0.004725</td>
      <td>-0.004152</td>
      <td>-0.000772</td>
      <td>-0.000764</td>
      <td>-0.000945</td>
      <td>-0.001716</td>
      <td>-0.001013</td>
      <td>-0.005245</td>
      <td>-0.003313</td>
      <td>-0.002685</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>-0.000169</td>
      <td>-0.000772</td>
      <td>1.000000</td>
      <td>-0.000154</td>
      <td>-0.000155</td>
      <td>-0.000110</td>
      <td>-0.000110</td>
      <td>-0.000746</td>
      <td>-0.000219</td>
      <td>-0.000109</td>
      <td>...</td>
      <td>-0.000673</td>
      <td>-0.000591</td>
      <td>-0.000110</td>
      <td>-0.000109</td>
      <td>-0.000135</td>
      <td>-0.000244</td>
      <td>-0.000144</td>
      <td>-0.001047</td>
      <td>-0.000472</td>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>-0.000238</td>
      <td>-0.001084</td>
      <td>-0.000154</td>
      <td>1.000000</td>
      <td>-0.000218</td>
      <td>-0.000154</td>
      <td>-0.000154</td>
      <td>-0.001048</td>
      <td>-0.000308</td>
      <td>-0.000153</td>
      <td>...</td>
      <td>-0.000945</td>
      <td>-0.000831</td>
      <td>-0.000154</td>
      <td>-0.000153</td>
      <td>-0.000189</td>
      <td>-0.000343</td>
      <td>-0.000203</td>
      <td>-0.001471</td>
      <td>-0.000663</td>
      <td>-0.000537</td>
    </tr>
    <tr>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <td>-0.000239</td>
      <td>-0.001091</td>
      <td>-0.000155</td>
      <td>-0.000218</td>
      <td>1.000000</td>
      <td>-0.000155</td>
      <td>-0.000155</td>
      <td>-0.001055</td>
      <td>-0.000310</td>
      <td>-0.000154</td>
      <td>...</td>
      <td>-0.000951</td>
      <td>-0.000836</td>
      <td>-0.000155</td>
      <td>-0.000154</td>
      <td>-0.000190</td>
      <td>-0.000346</td>
      <td>-0.000204</td>
      <td>-0.001480</td>
      <td>-0.000667</td>
      <td>-0.000541</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>-0.000376</td>
      <td>-0.001716</td>
      <td>-0.000244</td>
      <td>-0.000343</td>
      <td>-0.000346</td>
      <td>-0.000244</td>
      <td>-0.000244</td>
      <td>-0.001660</td>
      <td>-0.000487</td>
      <td>-0.000242</td>
      <td>...</td>
      <td>-0.001496</td>
      <td>-0.001315</td>
      <td>-0.000244</td>
      <td>-0.000242</td>
      <td>-0.000299</td>
      <td>1.000000</td>
      <td>-0.000321</td>
      <td>-0.002328</td>
      <td>-0.001049</td>
      <td>-0.000850</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>-0.000222</td>
      <td>-0.001013</td>
      <td>-0.000144</td>
      <td>-0.000203</td>
      <td>-0.000204</td>
      <td>-0.000144</td>
      <td>-0.000144</td>
      <td>-0.000980</td>
      <td>-0.000288</td>
      <td>-0.000143</td>
      <td>...</td>
      <td>-0.000884</td>
      <td>-0.000776</td>
      <td>-0.000144</td>
      <td>-0.000143</td>
      <td>-0.000177</td>
      <td>-0.000321</td>
      <td>1.000000</td>
      <td>-0.001375</td>
      <td>-0.000619</td>
      <td>-0.000502</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>-0.001611</td>
      <td>-0.005245</td>
      <td>-0.001047</td>
      <td>-0.001471</td>
      <td>-0.001480</td>
      <td>-0.001047</td>
      <td>-0.001047</td>
      <td>0.002321</td>
      <td>-0.002086</td>
      <td>-0.001036</td>
      <td>...</td>
      <td>0.003016</td>
      <td>-0.005632</td>
      <td>-0.001047</td>
      <td>-0.001036</td>
      <td>-0.001282</td>
      <td>-0.002328</td>
      <td>-0.001375</td>
      <td>1.000000</td>
      <td>-0.004494</td>
      <td>-0.003642</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>-0.000726</td>
      <td>-0.003313</td>
      <td>-0.000472</td>
      <td>-0.000663</td>
      <td>-0.000667</td>
      <td>-0.000472</td>
      <td>-0.000472</td>
      <td>-0.003203</td>
      <td>-0.000940</td>
      <td>-0.000467</td>
      <td>...</td>
      <td>-0.002888</td>
      <td>-0.002538</td>
      <td>-0.000472</td>
      <td>-0.000467</td>
      <td>-0.000578</td>
      <td>-0.001049</td>
      <td>-0.000619</td>
      <td>-0.004494</td>
      <td>1.000000</td>
      <td>-0.001641</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>-0.000588</td>
      <td>-0.002685</td>
      <td>-0.000382</td>
      <td>-0.000537</td>
      <td>-0.000541</td>
      <td>-0.000382</td>
      <td>-0.000382</td>
      <td>-0.002596</td>
      <td>-0.000762</td>
      <td>-0.000378</td>
      <td>...</td>
      <td>-0.002341</td>
      <td>-0.002057</td>
      <td>-0.000382</td>
      <td>-0.000378</td>
      <td>-0.000468</td>
      <td>-0.000850</td>
      <td>-0.000502</td>
      <td>-0.003642</td>
      <td>-0.001641</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>206 rows × 206 columns</p>
</div>






```python
item_corr_df3[item_corr_df3<0] = 0
common_item_pred_ratings = np.dot(item_corr_df3,common_item_pivot.fillna(0))
common_item_pred_ratings.shape
```




    (206, 8379)




```python
test_items = common_item.copy()

```


```python
test_item_tb = test_items.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating').T.fillna(0)
final_item_ratings = np.multiply(common_item_pred_ratings,test_item_tb)
final_item_ratings
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>reviews_username</th>
      <th>00dog3</th>
      <th>01impala</th>
      <th>08dallas</th>
      <th>09mommy11</th>
      <th>1143mom</th>
      <th>1234</th>
      <th>123charlie</th>
      <th>123numbers</th>
      <th>12cass12</th>
      <th>132457</th>
      <th>...</th>
      <th>zombiedad80</th>
      <th>zombiegirl22</th>
      <th>zombiekiller</th>
      <th>zoney86</th>
      <th>zookeeper</th>
      <th>zpalma</th>
      <th>zsarah</th>
      <th>zulaa118</th>
      <th>zxjki</th>
      <th>zzz1127</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100:Complete First Season (blu-Ray)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-2018 Brownline174 Duraflex 14-Month Planner 8 1/2 X 11 Black</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2x Ultra Era with Oxi Booster, 50fl oz</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4C Grated Parmesan Cheese 100% Natural 8oz Shaker</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>WeatherTech 40647 14-15 Outlander Cargo Liners Behind 2nd Row, Black</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Weleda Everon Lip Balm</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Windex Original Glass Cleaner Refill 67.6oz (2 Liter)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Yes To Carrots Nourishing Body Wash</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Yes To Grapefruit Rejuvenating Body Wash</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>206 rows × 8379 columns</p>
</div>




```python
X  = final_item_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)
y.shape
```

    MinMaxScaler(feature_range=(1, 5))
    [[nan nan nan ... nan nan nan]
     [nan nan  1. ... nan nan nan]
     [nan nan nan ... nan nan nan]
     ...
     [nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]
     [nan nan nan ... nan nan nan]]
    




    (206, 8379)




```python
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))
```


```python
rmse = (sum(sum((common_item_pivot -  y )**2))/total_non_nan)**0.5
print(rmse)
```

    3.5541206040111435
    

## Best suited recommendation system 
- By checking the root mean squared error for user-user and item-item recommendation systems . 


```
User-User recommendation
```

 will be used as it is having less RMSE

##### Final ```model.py``` attached which will recommend the top 20 products and then top 5 products based on sentiment 
##### pickle files are available in ```pickle_file``` folder
##### ```app.py``` contains the flask deployment 
##### Heroku URL : ```https://sentimentbasedrecommendationsy.herokuapp.com/```


