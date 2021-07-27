import pandas as pd 
import numpy as np 
import time 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests

#reading in the data
data = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')
#importing the stop words
stopwords = requests.get( "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" ).content.decode('utf-8').split( "\n" )

#starting the timer
start = time.time()

#task 1.1
#isolating the sentiment column of the data 
sentiments = data.loc[:,"Sentiment"]
#taking the previously isolated column and applying the set() function to get the unique attributes of the sentiment columns
possible_sentiments = set(sentiments)
#printing out the possible sentiments as asked for in q1.1 
print("The possible sentiments that a tweet may have are: ", possible_sentiments)
print(" ")
#using the .value_counts() function to count the intances of each possible sentiment and using the .iloc[1] to access the second most popular sentiment
second_most_popular_sentiment = (pd.DataFrame(sentiments.value_counts())).iloc[[1]]
#printing out the second most popular sentiment as asked for in q1.1
print("The second most popular sentiment is:",second_most_popular_sentiment, sep='\n')
print(" ")
#isolating the data of the columns that contain the date the tweet was made ("TweetAt") and the sentiment of the tweet
df = pd.DataFrame(data.loc[:,["TweetAt", "Sentiment"]])
#variable to contain the value we are looking for in this case the sentiment of extremely positive
xpositive_sentiment = "Extremely Positive"

#finding the day with the most extremely positive tweets
data_frame = df[df["Sentiment"] == xpositive_sentiment].groupby('TweetAt').count().rename(columns={"Sentiment":xpositive_sentiment}).nlargest(1, xpositive_sentiment)
#printing the date with the most extremely positive tweets as asked for in q1.1
print("The date with the most extremely positive tweets was on: ",data_frame, sep='\n')
print("")

#task 1.2 
#code to clean the data as asked for. Firstly i make all the characters lower case as asked for, secondly i remove the urls as i feel these qualify as symbols
#and dont contribute to the meaning of the tweet. thirdly i remove the symbols as asked for, i then remove all the additional white space larger than one space. 
#finally i strip the data to remove any addtional excess white space. 
data['OriginalTweet'] = data['OriginalTweet']\
                        .str.lower()\
                        .str.replace(r"http?\S+", ' ', regex=True)\
                        .str.replace(r'[^a-zA-Z]', r' ', regex=True)\
                        .str.replace(r'\s+', ' ', regex=True)\
                        .str.strip()
#splitting the above data so that it is tokenized and i can use .value_counts() to count all the instances of all the words
tweets = data['OriginalTweet'].str.split(' ')
#using .value_counts() to count all the words
tweets_un_formatted = pd.Series(np.concatenate(tweets)).value_counts()

#creating a set of stopwords just to make sure that there arent duplicates or irregularities
set_of_stop_words = set(stopwords)
no_stop_words = data['OriginalTweet']
#removing words with 2 chars or less and removing addtional spaces. 
no_stop_words = no_stop_words\
                    .str.replace(r'\b(\w{1,2})\b', ' ')\
                    .str.replace(r'\s+', ' ', regex=True)
#splitting the data so that its tokenized
no_stop_words = no_stop_words.str.split(' ')   
#removing the stopwords 
no_stop_words = no_stop_words.apply(lambda x: [word for word in x if word not in set_of_stop_words])    
#using .value_counts() to count all the words in the newly constructed data set without stop words 
tweets_formatted = pd.Series(np.concatenate(no_stop_words)).value_counts()

#printing out all the values that are asked for in q1.2
print("Total number of words including repetitions in the corpus of unformatted tweets: ", tweets_un_formatted.sum(axis=0))
print("Total number of distinct words in the corpus of unformatted tweets:", len(tweets_un_formatted.index))
print("The 10 most popular words in the corpus of unformatted tweets:", tweets_un_formatted.iloc[:10], sep='\n')
print(" ")
print("Total number of words including repetitions in the corpus of formatted tweets: ", tweets_formatted.sum(axis=0))
#print("Total number of distinct words in the corpus of formatted tweets:", len(tweets_formatted.index))
print("The 10 most popular words in the corpus of formatted tweets:", tweets_formatted.iloc[:10], sep='\n')
print(" ")

#task 1.3
#to plot the histogram we first have to remove all the duplicates in the tweets 
#using the data constructed above where the stop words have been removed and then i remove all the duplicates 
#this ensures that we get an accurate representation of which words are in which specific documents or in this case tweets
histogram_data = no_stop_words.map(set).str.join(' ').str.split(' ')

#here we use .value_counts() to count the amount of tweets in which each word occurs
final_histogram_data = pd.Series(np.concatenate(histogram_data)).value_counts(ascending=True)
#we plot the range of the words we have counted above as the x axis and the values of each word divided by the number of all tweets to ensure the fraction of how many
#tweets each word is in is represented
plt.plot(range(len(final_histogram_data)), (final_histogram_data.values))
#this line is commented out so that the original histogram is show as wanted by it can be uncommented to show the y axis in log scale to make the graph easier to read
#plt.yscale('log')
plt.title('Normal Histogram')
plt.xlabel('Number of the word used to represent it in the corpus')
plt.ylabel('Fraction of documents in which a word occurs')


#task 1.4 
#here i take the orignal tweet data and the sentiment data and convert them to numpy arrays 
X_data = data["OriginalTweet"].to_numpy()
y = data["Sentiment"].to_numpy()
#initializing the countVectorizer 
vectorizer = CountVectorizer()
#transforming the tweet data with the countVectorizer so that we can use it with the classfier
X = vectorizer.fit_transform(X_data)
#initializing the multinomial bayes classifier 
clf = MultinomialNB()
#fitting the classifier with the data that has been transformed with the countVectorizer and the sentiment data
clf.fit(X, y)
#error rate calculation of the classifier
error_rate = 1 - clf.score(X, y)
#printing the error rate
print("The error rate of the classifier is: ",error_rate, "or", (error_rate*100), "%")
print(" ")
#stopping the timer
end = time.time()
#printing the time taken by the program to run
print("Total run time of the program is: " ,end - start, "seconds")
#showing the histogram, this is done here because otherwise it would make the runtime of the program as long as the histogram is displayed. 
plt.show()
