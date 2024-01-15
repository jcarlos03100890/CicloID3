#import pandas as pd
from a_auxiliar_funtions import *

#Get data from database
df = get_data_from_data_base("tweets")

# Checking the data type
print("\n\nChecking the data type\n")
print(df.dtypes)

print("\n\nTotal number of rows and columns loaded from the database:",df.shape)

#Cleanig the tweet text into tweet field
df['tweet'] = df['texto'].apply(clean_tweet)

#Create two new columns ‘Subjectivity’ & ‘Polarity’
df['Subjectivity'] = df['tweet'].apply(getSubjectivity)
df['Polarity'] = df['tweet'].apply(getPolarity)

#Create a new column 'Sentiment'
df['sentiment'] = df['Polarity'].apply(getAnalysis )

#Get sentiment analysis using VADER SentimentIntensityAnalyzer
df['neg'] = df['tweet'].apply(get_neg_score)
df['pos'] = df['tweet'].apply(get_pos_score)
df['neu'] = df['tweet'].apply(get_neu_score)
df['compound'] = df['tweet'].apply(get_compound_score)

#Create a new column 'Sentiment'
df['vader_sentiment'] = df['compound'].apply(get_vader_Analysis )

#Get the data after sentiment analysis
data=df[['tweet','Subjectivity','Polarity','sentiment','neg','pos','neu','compound','vader_sentiment']]

data_by_textblob = data_text = data.groupby("sentiment")["sentiment"].count()
data_by_vader = data_text = data.groupby("vader_sentiment")["vader_sentiment"].count()

print("\n\nCounts using TextBlob sentiment analyzer:\n",data_by_textblob)
print("\n\nCounts using VADER sentiment analyzer:\n",data_by_vader)

#Set the data into the database
# Checking the data type
print("\n\nChecking the data type\n")
print(data.dtypes)

print("\n\nTotal number of rows and columns:",data.shape)

print("\n\nTo display the top 10 rows\n")
print(data.head(10))

insert_table("twitter_sentiment",data)
