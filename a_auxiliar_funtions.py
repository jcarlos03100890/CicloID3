from sqlalchemy import create_engine
import pandas as pd
import re
from textblob import TextBlob
import nltk
from unicodedata import normalize
import constant
from nltk.sentiment import SentimentIntensityAnalyzer


#Create a funcion to get the data from database
def get_data_from_data_base(table_name):
        
    # Step 1: Create a SQLAlchemy Engine to Connect to MySQL
    engine = create_engine(f"mysql+mysqlconnector://{constant.db_user}:{constant.db_password}@{constant.db_host}:{constant.db_port}/{constant.db_name}")

    #Step 2: Read from the database 
    df_results = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

    #Step 3: Close the Database Connection
    engine.dispose()

    return df_results


#Create a function to storage the data into database
def insert_table(table_name,df):

    # Step 1: Create a SQLAlchemy Engine to Connect to MySQL
    engine = create_engine(f"mysql+mysqlconnector://{constant.db_user}:{constant.db_password}@{constant.db_host}:{constant.db_port}/{constant.db_name}")

    # Step 3: Write DataFrame to MySQL
    df.to_sql(table_name, con=engine, index=False, if_exists='replace',chunksize = 50000)  # Change 'replace' to 'append' if needed

    # Step 3: Close the Database Connection
    engine.dispose()

    
#Create a function to cleaning the text
def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    '''

    #Set lowercase the text
    tweet = tweet.lower()
    #Replacing the especial characters
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    tweet = normalize('NFKC', normalize('NFKD', tweet).translate(trans_tab))
    #Cleaning and removing the stop words from the tweet text
    stopword = nltk.corpus.stopwords.words('spanish')
    tweet = " ".join([word for word in str(tweet).split() if word not in stopword])
    #Cleaning and removing repeating characters
    tweet = re.sub(r'(.)1+', r'1', tweet)
    #Cleaning and removing URLs, Numbers, letting the ñ
    
    return " ".join(re.sub("(\w+:\/\/\S+)|([^0-9A-Za-zñ \t])|[0-9]","",tweet).split())

#Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
  
#Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Get a function to get sentiment value
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

#Create the fucntions to use VADER Sentiment analysis
#Get 'neg' score
def get_neg_score(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['neg']

#Get 'pos' score
def get_pos_score(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['pos']

#Get 'neu' score
def get_neu_score(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['neu']

#Get 'compound' score
def get_compound_score(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

#Get a function to get sentiment value
def get_vader_Analysis(score):
    if score >= 0.05 :
        return 'Positive'
    elif score  <= - 0.05:
        return 'Negative'
    else:
        return 'Neutral'

