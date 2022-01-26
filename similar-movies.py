#!/usr/bin/env python
# coding: utf-8

# ## 1. Import and observe dataset
# <p>We all love watching movies! There are some movies we like, some we don't. Most people have a preference for movies of a similar genre. Some of us love watching action movies, while some of us like watching horror. Some of us like watching movies that have ninjas in them, while some of us like watching superheroes.</p>
# <p>Movies within a genre often share common base parameters. Consider the following two movies:</p>
# <p><img style="margin:5px 20px 5px 1px; height: 250px; display: inline-block;" alt="2001: A Space Odyssey" src="https://assets.datacamp.com/production/project_648/img/movie1.jpg">
# <img style="margin:5px 20px 5px 1px; height: 250px; display: inline-block;" alt="Close Encounters of the Third Kind" src="https://assets.datacamp.com/production/project_648/img/movie2.jpg"></p>
# <p>Both movies, <em>2001: A Space Odyssey</em> and <em>Close Encounters of the Third Kind</em>, are movies based on aliens coming to Earth. I've seen both, and they indeed share many similarities. We could conclude that both of these fall into the same genre of movies based on intuition, but that's no fun in a data science context. In this notebook, we will quantify the similarity of movies based on their plot summaries available on IMDb and Wikipedia, then separate them into groups, also known as clusters. We'll create a dendrogram to represent how closely the movies are related to each other.</p>
# <p>Let's start by importing the dataset and observing the data provided.</p>

# In[1]:


# Import modules
import numpy as np
import pandas as pd
import nltk

# Set seed for reproducibility
np.random.seed(5)

# Reading the dataset
movies_df = pd.read_csv("movies.csv")

print("Number of movies loaded: %s " % (len(movies_df)))

movies_df


# ## 2. Combine Wikipedia and IMDb plot summaries
# <p>The dataset we imported currently contains two columns titled <code>wiki_plot</code> and <code>imdb_plot</code>. They are the plot found for the movies on Wikipedia and IMDb, respectively. The text in the two columns is similar, however, they are often written in different tones and thus provide context on a movie in a different manner of linguistic expression. Further, sometimes the text in one column may mention a feature of the plot that is not present in the other column. For example, consider the following plot extracts from <em>The Godfather</em>:</p>
# <ul>
# <li>Wikipedia: "On the day of his only daughter's wedding, Vito Corleone"</li>
# <li>IMDb: "In late summer 1945, guests are gathered for the wedding reception of Don Vito Corleone's daughter Connie"</li>
# </ul>
# <p>While the Wikipedia plot only mentions it is the day of the daughter's wedding, the IMDb plot also mentions the year of the scene and the name of the daughter. </p>
# <p>Let's combine both the columns to avoid the overheads in computation associated with extra columns to process.</p>

# In[2]:


# Combine wiki_plot and imdb_plot into a single column
movies_df["plot"] = movies_df["wiki_plot"].astype(str) + "\n" +                  movies_df["imdb_plot"].astype(str)

movies_df.head()


# ## 3. Tokenization
# <p>Tokenization is the process  by which we break down articles into individual sentences or words, as needed. Besides the tokenization method provided by NLTK, we might have to perform additional filtration to remove tokens which are entirely numeric values or punctuation.</p>
# <p>While a program may fail to build context from "While waiting at a bus stop in 1981" (<em>Forrest Gump</em>), because this string would not match in any dictionary, it is possible to build context from the words "while", "waiting" or "bus" because they are present in the English dictionary. </p>
# <p>Let us perform tokenization on a small extract from <em>The Godfather</em>.</p>

# In[3]:


# Tokenize a paragraph
sent_tokenized = [sent for sent in nltk.sent_tokenize("""
                        Today (May 19, 2016) is his only daughter's wedding.
                        Vito Corleone is the Godfather.
                        """)]

# Word Tokenize first sentence from sent_tokenized, save as words_tokenized
words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]

import re

filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

filtered


# ## 4. Stemming
# <p>Stemming is the process by which we bring down a word from its different forms to the root word. This helps us establish meaning to different forms of the same words without having to deal with each form separately. For example, the words 'fishing', 'fished', and 'fisher' all get stemmed to the word 'fish'.</p>
# <p>Consider the following sentences:</p>
# <ul>
# <li>"Young William Wallace witnesses the treachery of Longshanks" ~ <em>Gladiator</em></li>
# <li>"escapes to the city walls only to witness Cicero's death" ~ <em>Braveheart</em></li>
# </ul>
# <p>Instead of building separate dictionary entries for both witnesses and witness, which mean the same thing outside of quantity, stemming them reduces them to 'wit'.</p>
# <p>There are different algorithms available for stemming such as the Porter Stemmer, Snowball Stemmer, etc. We shall use the Snowball Stemmer.</p>

# In[4]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

print("Without stemming: ", filtered)

stemmed_words = [stemmer.stem(t) for t in filtered]

print("After stemming:   ", stemmed_words)


# ## 5. Club together Tokenize & Stem
# <p>We are now able to tokenize and stem sentences. But we may have to use the two functions repeatedly one after the other to handle a large amount of data, hence we can think of wrapping them in a function and passing the text to be tokenized and stemmed as the function argument. Then we can pass the new wrapping function, which shall perform both tokenizing and stemming instead of just tokenizing, as the tokenizer argument while creating the TF-IDF vector of the text.  </p>
# <p>What difference does it make though? Consider the sentence from the plot of <em>The Godfather</em>: "Today (May 19, 2016) is his only daughter's wedding." If we do a 'tokenize-only' for this sentence, we have the following result:</p>
# <blockquote>
#   <p>'today', 'may', 'is', 'his', 'only', 'daughter', "'s", 'wedding'</p>
# </blockquote>
# <p>But when we do a 'tokenize-and-stem' operation we get:</p>
# <blockquote>
#   <p>'today', 'may', 'is', 'his', 'onli', 'daughter', "'s", 'wed'</p>
# </blockquote>
# <p>All the words are in their root form, which will lead to a better establishment of meaning as some of the non-root forms may not be present in the NLTK training corpus.</p>

# In[5]:


# Define a function for stemming and tokenization
def tokenize_and_stem(text):

    # Tokenize by sentence, then by word
    sents = [s for s in nltk.sent_tokenize(text)]
    tokens = [word for word in nltk.word_tokenize(sents[0])]

    # Filter out raw tokens
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    stems = [stemmer.stem(ft) for ft in filtered_tokens]

    return stems

words_stemmed = tokenize_and_stem("Today (May 19, 2016) is his only daughter's wedding.")
print(words_stemmed)


# ## 6. Create TfidfVectorizer
# <p>Computers do not <em>understand</em> text. These are machines only capable of understanding numbers and performing numerical computation. Hence, we must convert our textual plot summaries to numbers for the computer to be able to extract meaning from them. One simple method of doing this would be to count all the occurrences of each word in the entire vocabulary and return the counts in a vector. Enter <code>CountVectorizer</code>.</p>
# <p>Consider the word 'the'. It appears quite frequently in almost all movie plots and will have a high count in each case. But obviously, it isn't the theme of all the movies! <a href="https://campus.datacamp.com/courses/natural-language-processing-fundamentals-in-python/simple-topic-identification?ex=11">Term Frequency-Inverse Document Frequency</a> (TF-IDF) is one method which overcomes the shortcomings of <code>CountVectorizer</code>. The Term Frequency of a word is the measure of how often it appears in a document, while the Inverse Document Frequency is the parameter which reduces the importance of a word if it frequently appears in several documents.</p>
# <p>For example, when we apply the TF-IDF on the first 3 sentences from the plot of <em>The Wizard of Oz</em>, we are told that the most important word there is 'Toto', the pet dog of the lead character. This is because the movie begins with 'Toto' biting someone due to which the journey of Oz begins!</p>
# <p>In simplest terms, TF-IDF recognizes words which are unique and important to any given document. Let's create one for our purposes.</p>

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# TfidfVectorizer object with stopwords
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200000,
                                 min_df=0.15, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))


# ## 7. Fit transform TfidfVectorizer
# <p>Once we create a TF-IDF Vectorizer, we must fit the text to it and then transform the text to produce the corresponding numeric form of the data which the computer will be able to understand and derive meaning from. To do this, we use the <code>fit_transform()</code> method of the <code>TfidfVectorizer</code> object. </p>
# <p>If we observe the <code>TfidfVectorizer</code> object we created, we come across a parameter stopwords. 'stopwords' are those words in a given text which do not contribute considerably towards the meaning of the sentence and are generally grammatical filler words. For example, in the sentence 'Dorothy Gale lives with her dog Toto on the farm of her Aunt Em and Uncle Henry', we could drop the words 'her' and 'the', and still have a similar overall meaning to the sentence. Thus, 'her' and 'the' are stopwords and can be conveniently dropped from the sentence. </p>
# <p>On setting the stopwords to 'english', we direct the vectorizer to drop all stopwords from a pre-defined list of English language stopwords present in the nltk module. Another parameter, <code>ngram_range</code>, defines the length of the ngrams to be formed while vectorizing the text.</p>

# In[7]:


# Fit and transform tfidf_vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])

print(tfidf_matrix.shape)


# ## 8. Import KMeans and create clusters
# <p>To determine how closely one movie is related to the other by the help of unsupervised learning, we can use clustering techniques. Clustering is the method of grouping together a number of items such that they exhibit similar properties. According to the measure of similarity desired, a given sample of items can have one or more clusters. </p>
# <p>A good basis of clustering in our dataset could be the genre of the movies. Say we could have a cluster '0' which holds movies of the 'Drama' genre. We would expect movies like <em>Chinatown</em> or <em>Psycho</em> to belong to this cluster. Similarly, the cluster '1' in this project holds movies which belong to the 'Adventure' genre (<em>Lawrence of Arabia</em> and the <em>Raiders of the Lost Ark</em>, for example).</p>
# <p>K-means is an algorithm which helps us to implement clustering in Python. The name derives from its method of implementation: the given sample is divided into <b><i>K</i></b> clusters where each cluster is denoted by the <b><i>mean</i></b> of all the items lying in that cluster. </p>
# <p>We get the following distribution for the clusters:</p>
# <p><img src="https://assets.datacamp.com/production/project_648/img/bar_clusters.png" alt="bar graph of clusters"></p>

# In[8]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=5)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# Create a column cluster
movies_df["cluster"] = clusters

# Number of movies per cluster
movies_df['cluster'].value_counts()


# ## 9. Calculate similarity distance
# <p>Consider the following two sentences from the movie <em>The Wizard of Oz</em>: </p>
# <blockquote>
#   <p>"they find in the Emerald City"</p>
#   <p>"they finally reach the Emerald City"</p>
# </blockquote>
# <p>If we put the above sentences in a <code>CountVectorizer</code>, the vocabulary produced would be "they, find, in, the, Emerald, City, finally, reach" and the vectors for each sentence would be as follows: </p>
# <blockquote>
#   <p>1, 1, 1, 1, 1, 1, 0, 0</p>
#   <p>1, 0, 0, 1, 1, 1, 1, 1</p>
# </blockquote>
# <p>When we calculate the cosine angle formed between the vectors represented by the above, we get a score of 0.667. This means the above sentences are very closely related. <em>Similarity distance</em> is 1 - <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine similarity angle</a>. This follows from that if the vectors are similar, the cosine of their angle would be 1 and hence, the distance between then would be 1 - 1 = 0.</p>
# <p>Let's calculate the similarity distance for all of our movies.</p>

# In[9]:


# Import cosine_similarity to calculate similarity of movie plots
from sklearn.metrics.pairwise import cosine_similarity

# Similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)


# ## 10. Import Matplotlib, Linkage, and Dendrograms
# <p>We shall now create a tree-like diagram (called a dendrogram) of the movie titles to help us understand the level of similarity between them visually. Dendrograms help visualize the results of hierarchical clustering, which is an alternative to k-means clustering. Two pairs of movies at the same level of hierarchical clustering are expected to have similar strength of similarity between the corresponding pairs of movies. For example, the movie <em>Fargo</em> would be as similar to <em>North By Northwest</em> as the movie <em>Platoon</em> is to <em>Saving Private Ryan</em>, given both the pairs exhibit the same level of the hierarchy.</p>
# <p>Let's import the modules we'll need to create our dendrogram.</p>

# In[10]:


import matplotlib.pyplot as plt

# Configure matplotlib to display the output inline
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.hierarchy import linkage, dendrogram


# ## 11. Create merging and plot dendrogram
# <p>We shall plot a dendrogram of the movies whose similarity measure will be given by the similarity distance we previously calculated. The lower the similarity distance between any two movies, the lower their linkage will make an intercept on the y-axis. For instance, the lowest dendrogram linkage we shall discover will be between the movies, <em>It's a Wonderful Life</em> and <em>A Place in the Sun</em>. This indicates that the movies are very similar to each other in their plots.</p>

# In[11]:


# Mergings matrix
mergings = linkage(similarity_distance, method='complete')

# Plot dendrogram
dendrogram_ = dendrogram(mergings,
               labels=[x for x in movies_df["title"]],
               leaf_rotation=90,
               leaf_font_size=16,)

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

plt.show()


# ## 12. Which movies are most similar?
# <p>We can now determine the similarity between movies based on their plots! To wrap up, let's answer one final question: which movie is most similar to the movie <em>Braveheart</em>?</p>

# In[344]:


# Answer the question
ans = "Gladiator"
print(ans)
