

# **How Text Embeddings help suggest similar words**
The world is filled with fascinating technology. It is overwhelming to see such extravagant machines, but we tend to forget the engineering that goes behind our most trivial tasks . An example of this is our unofficial right-hand man: our smartphone. We often just mindlessly do our tasks or scroll through social media without even realizing how incredible the procedure behind it is. The thorough applications of **Natural Language Processing (NLP)** and **Machine Learning (ML)** in our smartphones, such as personal assistant apps like Amazon Alexa and Google Translate, GPS navigation apps and spam filtering and even the auto-correction as we write this blog, are hidden in plain sight.

  

Significant advancements in the field have been achieved because of Text Embeddings, which addressed some key challenges faced in representing words, sentences, and documents in ways that machines can understand and prove to be highly efficient and accurate. 

Let's dive deeper into how to use Text Embeddings to help suggest similar words!

## **What are text embeddings?**
Text embeddings are a way to represent words or textual documents as large dimensional mathematical vectors which allows our dear “dumb” computers to understand and process them more efficiently. They transform words into numerical vectors which capture their meaning based on their **context.** In this way, words that share similar meanings, context, or analogies are placed close together in the **embedding space.** Here’s a highly simplified example of how the words are represented and what counts as being close to each other.

![](https://i.imgur.com/caRaOyN.png)

Here we can see that the words “king” and “queen” are placed together in the space, whereas “bartender” is away from them, indicating no strong connection. Although this diagram only portrays 2-D vectors, in actual applications, the vectors are much larger (even upto 100 dimensions), and this is in fact a great advantage since let’s say a vector holding 100 values, is able to represent 100 different features!

Another key usage of these embeddings is to handle polysemic words, which are the same sounding/written words having different meanings as per the context. For example, consider these two sentences :-

*“I ate an apple.” vs. “Apple released a new phone this year.”*

Here the word “apple” is being used in a different context. Text embeddings are able to understand this and suggest contextually accurate alternatives.

But what was the need for them? What did the earlier, so called “traditional” models lack that the development of new methods was required? Let’s discuss the major issues and problems that the old models faced, and how text embeddings overcame them.

### **Issues with the traditional models**
The traditional models such as the Bag Of Words (BOW) or the Term-Frequency Inverse Document Frequency (TF-IDF) models relied on trivial metrics such as the count of each word and term frequencies. This way, the words are represented as individual entities, hence they lose important properties such as ordering, contextual information, analogous words and sparsity due to high dimensionality.

Text Embeddings models such as Word2Vec, GLoVe, and BERT, represent words as dense high dimensional vectors. This way, similar words can be represented closer to each other and the context of the words is captured. For example the words “happy” and “joyful” are closer in the embedding space. Also the **word ordering is preserved**. Sentences such as “The man bit the dog” vs “The dog bit the man” are interpreted differently (as they should be), while the earlier models would capture no difference between them.

Speaking about the traditional models, let’s dive in a bit in the most used earlier models, The Bag Of Words and the Term Frequency - Inverse Document Frequency.

## **Bag of words**
Before we start digging into them, let’s set the problem that we will be solving throughout this blog. You will be working with word suggestions for search engines. This is a really important problem for the field of Search Engine Optimization (SEO).

In the popular Bag Of Words Model, we **vectorize words** based on their count in the document or sample. Here’s how you can build a BOW :-

-   You remove punctuations and lower the case
    
-   Then you eliminate the stopwords (words that are not meaningful for the suggestion, eg:- “and”, “or”, “the” etc.)
    
-   After this you create the count vector using different libraries, and then apply your models.
    

Now let’s see this in action! The corpus is a small set of search queries of buying electronics :-

    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk
    search_queries = [
    "buy laptop online","cheap laptops for sale","best gaming laptop","online shopping for electronics",
    "buy smartphone online","cheap mobile phones","best budget smartphone","latest smartphone models",
    "buy books online","top books to read","best fantasy books"
    ]
    
    corpus = []  #processing the corpus
    for query in search_queries:
       review = query.lower()
       review = review.split()
       review = [word for word in review if word not  in  set(sw)]
       review = ' '.join(review)
       corpus.append(review)

    #vectorization of BOW
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(corpus)
    X_bow_dense = X_bow.toarray()
    df_bow = pd.DataFrame(X_bow_dense, columns=vectorizer_bow.get_feature_names_out())
    df_bow.insert(0,  'Index',  range(len(corpus)))
    print("Bag of Words (BOW) Table:")
    print(df_bow.to_string(index=False))

| Index | best | books | budget | buy | cheap | electronics | fantasy | gaming | laptop | laptops | latest | mobile | models | online | phones | read | sale | shopping | smartphone | top |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 
| 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 5 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 6 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 8 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| 10 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |



## **Term Frequency - Inverse Document Frequency (TF-IDF)**
This algorithm works on the statistical principle of finding the **relevance of the word** in a document or a set of documents.

The term frequency (TF) score measures the **frequency of a word** occurring in the document while the inverse document frequency (IDF) measures the **rarity of the words** in the corpus. It is given more mathematical importance as some words rarely occurring in the text still might hold relevant information.

Now let’s see the expressions that are used to calculate the tf-idf score:-
***tf-idf*** $_{i,j}=Term \; Frequency_{i,j} \times Inverse \; Document \; Frequency_{i}$

Where,
***Term Frequency*** $_{i,j}=\frac{Term\;i\;frequency\;in\;document\;j}{Total\;no.\;of\;terms\;in\;document\;j}$

***Inverse Document Frequency*** $_{i,j}=log(\frac{Total\;documents}{No.\;of\;documents\;containing\;term\;i})$

Here’s the representation through a code snippet:-

    #tf-idf representation
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(corpus)
    X_tfidf_dense = X_tfidf.toarray()
    df_tfidf = pd.DataFrame(X_tfidf_dense, columns=vectorizer_tfidf.get_feature_names_out())
    df_tfidf.insert(0, 'Index', range(len(corpus)))
    print("\nTF-IDF Table:")
    print(df_tfidf.to_string(index=False))

*TF-IDF Table :*

![](https://i.imgur.com/W64hXDY.jpeg)

## **How are embeddings different**
In this section you will see two of the most popular models that are currently being used in text embeddings, which are the Word2Vec model and the GloVe model.

### **Word2Vec**
The Word2Vec model is a highly advanced machine learning model, currently being used for nearly all Natural Language Processing applications. It was developed by **Google** in 2013. The model works through two primary methods: **Continuous Bag of Words (CBOW)** and **Skip-gram**. These methods enable Word2Vec to learn word associations and place words with similar meanings close to each other in the vector space. This concept is often summarized by the phrase 	
***"You shall know a word by the company it keeps."***

The main architecture of the model can be associated to two methods as mentioned above:-

**Continuous Bag Of Words (CBOW):**

Here, instead of just using the count of each word, we define a sliding window which picks a target word, and its surrounding words are the context. For example:-
![](https://i.imgur.com/tE8Ge5B.png)
Here the word “pipelines” is the target word, whereas the other surrounding words explain the context.

The CBOW uses a **simple neural network** to process the probabilities of the suggestion. This includes an **input layer**, a single fully connected hidden layer also known as the **projection layer**, and an **output layer**. Here’s a simplified diagram of the neural network used by the CBOW method:-
![](https://i.imgur.com/yocbYgj.png)

**Skip-gram**
This method works **inversely** to the CBOW, where the **target word is known** and the model tries to guess the context using it. It is more effective in identifying less frequent relationships. Here’s a diagram for the Skip-gram method:-
![](https://i.imgur.com/KTPm0Vc.png)

    import gensim.downloader as api
    model = api.load("word2vec-google-news-300")

    example_word = "computer"
    similar_words = model.most_similar(example_word, topn=10)
    print(f"Top 10 words similar to '{example_word}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
*Output :*

![](https://i.imgur.com/cFAfH0s.jpeg)

![](https://i.imgur.com/nPweGEP.jpeg)

### GloVe

The GloVe method for text embeddings was developed at **Stanford by Jeffrey Pennington** and others. It is referred to as **global vectors** because the entire global corpus statistics were captured directly by the model. This helps it in finding great applications in word analogies. Unlike Word2Vec which relies on local contextual windows, the GloVe model relies on the global statistical parameters from a large corpus.

![](https://i.imgur.com/i7eG7bx.jpeg)

For its architecture, it constructs a **co-occurrence matrix**, which captures how frequently pairs of words appear together across the entire corpus. This matrix is then used to capture the semantic relations between the words.

Let’s see an example of a co-occurrence matrix in action :-

    # sample corpus
    corpus = [
    "I like pathway","I like NLP","I enjoy pathway",
    "deep learning is fun","NLP is amazing","pathway is fun"
    ]

![](https://i.imgur.com/aeIDOqL.jpeg)

**WINDOW SIZE 2**
Here the dimensions of the matrix are (m x m) if the number of unique words are m. Since the window size is selected as 2, it would count the co-occurrence in a window size of two words apart. Focusing on the word co-occurrence, it is able to derive much more complex and deeper relations between similar words.

Let us now show you the actual power of the GloVe model with some handy code:-

    glove_file = 'glove.6B.100d.txt'
    glove_model = load_glove_model(glove_file)
    def  find_similar_words(word, model, topn=10):
         word_vector = model[word]
         similarities = {}
         for other_word, other_vector in model.items():
             if other_word != word:
                similarity = cosine_similarity([word_vector],  [other_vector])[0][0]
                similarities[other_word] = similarity
         similar_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:topn]
         return similar_words
    example_word = "lovely"
    similar_words_glove = find_similar_words(example_word, glove_model, topn=10)
    print(f"Top 10 words similar to '{example_word}' (GloVe):")
    for word, similarity in similar_words_glove:
        print(f"{word}: {similarity:.4f}")

*Output :*

![**Output for the above code**](https://i.imgur.com/hwMVi6e.jpeg)

## **How text embeddings capture analogies**
Let us now try to understand how Text Embeddings capture the contextual relationships between countries and their capitals or verbs and tenses using GloVe embeddings, t-SNE, and PCA for visualization.

The fundamental approach of Text Embeddings is that we can represent **words as vectors** where every word can be expressed as a **numerical vector in a high-dimensional space**. Let's take the word 'king' for example. We can represent 'king' in an n-dimensional space where it will have n attributes. Words that appear in a similar context to 'king' will be **closer** to it in the embedding space. You can perform certain vector operations to reveal relationships between different words. The relationships can be shown as follows:-
1.  **Countries and Capitals :** The interrelation between Paris and France is analogous to the interrelation between Berlin and Germany. Subtracting France from Paris isolates the concept of a "capital city" concerning a country. The relation between the four words in the vector space can be presented as follows:

	*vec(Paris) − vec(France)≈vec(Germany) - vec(Berlin)*

	Hence, we can say, '"Paris is to France as Berlin is to Germany".

2. **Verb Tenses :** Verb Tenses can also be shown similarly. You can consider "Walking is to Walked" as "Running is to Ran". In the vector space, it is represented as follows:

	*Walking – Walked ≈ Running – Ran*

3. **Gender Analogy :** Construct a relationship between the words 'king' and 'queen'. In the embedding space, it can be expressed as follows:

	*vec(king)−vec(man)+vec(woman)≈vec(queen)*

	In the above expression, you can understand it as when you remove "man" from "king", it leaves a royal element, and when you add "woman", it becomes "queen".
- *Gender Analogy Suggestions code :*

		import gensim.downloader as api
		model = api.load("word2vec-google-news-300")
		result = model.most_similar(positive=['king',  'woman'], negative=['man'])
		for i in result:
			  print(f"{i[0]:<{20}}  {i[1]:.6f}")`
			  

-  *Output :*

	![Output	](https://i.imgur.com/OmJCFp3.jpeg)

- *PCA Plots :*

	![](https://i.imgur.com/M0vVuao.jpeg)

## Conclusion
Text embeddings are a way to represent words or textual documents as large dimensional mathematical vectors. They transform words into numerical vectors which capture their meaning based on their **context.**

A walk through the blog is as follows:

 - *Bag of Words* : vectorize words based on their count in the document or sample
 - *Term Frequency - Inverse Document Frequency (TF-IDF)* : measures the frequency of a word occurring in the document while the inverse document frequency (IDF) measures the rarity of the words in the corpus
 - *Word2Vec* : learn word associations and place words with similar meanings close to each other in the vector space
 - *Continuous Bag Of Words (CBOW)* : picks a target word, and its surrounding words are the context
 - *Skip-gram* : target word is known and the model tries to guess the context using it
 - *GloVe* : constructs a co-occurrence matrix which is 				then used to capture the semantic relations between the words

In this blog we discussed how the text embeddings capture the analogies using simple examples like Countries and Capitals, Verb Tenses and Gender Analogy. Now you know better how Text Embeddings work in hidden, plain sight.  

## **Citations**
- [IBM's blog on embeddings](https://www.ibm.com/topics/embedding)
- [Turing's blog on word embeddings](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp)
- [Stanford University's Global Vector Representations](https://nlp.stanford.edu/projects/glove/)
- [Pathway's Repositories](https://github.com/pathwaycom/pathway)

## **Authors**
**Team :**  **LLM-ited AI-dition**
Yashasvee Taiwade - *IIT Bombay*
Sajjad Nakhwa - *IIT Bombay*
