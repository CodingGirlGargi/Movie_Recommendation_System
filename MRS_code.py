import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
#print(movies)
#print(movies.shape)
credit = pd.read_csv("tmdb_5000_credits.csv")
#print(credits)
movies = movies.merge(credit,on = "title")
#print(movies.shape)

movies = movies[["movie_id", "title","overview","genres","keywords","cast","crew" ]]
#print(movies)
movies= movies.dropna()
#print(movies.shape)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

movies["genres"]=movies['genres'].apply(convert)
#print(movies)
movies["keywords"] = movies["keywords"].apply(convert)
#print(movies)

def castConvert(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i["name"])
            counter+=1
        else:
            break
    return l

movies["cast"] = movies["cast"].apply(castConvert)
#print(movies)


def fetchDir(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L
movies["crew"] = movies["crew"].apply(fetchDir)
#print(movies)
#print(movies["overview"][0])


movies["overview"] = movies["overview"].apply(lambda x: x.split())
#print(movies)
movies["genres"]= movies["genres"].apply(lambda x: [i.replace(" ","")for i in x])
#print(movies)


movies["keywords"]= movies["keywords"].apply(lambda x: [i.replace(" ","")for i in x])
movies["cast"]= movies["cast"].apply(lambda x: [i.replace(" ","")for i in x])
movies["crew"]= movies["crew"].apply(lambda x: [i.replace(" ","")for i in x])
#print(movies)

movies["tags"] = movies["overview"]+ movies["genres"]+ movies["keywords"]+ movies["cast"]+ movies["crew"]
#print(movies)

new_df = movies[["movie_id","title","tags"]]
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
#print(new_df)
#print(new_df['tags'][0])

new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())
#print(new_df['tags'][1])


ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df["tags"] = new_df["tags"].apply(stem)

#print(new_df)
cv = CountVectorizer(max_features = 5000, stop_words= "english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
#print(vectors)

#print(cosine_similarity(vectors).shape)
similarity = cosine_similarity(vectors)
availableList = (list(movies['title'].values))
availableList.sort()
print("You can choose a movie from: ")
print(availableList)


def recommend(movie):
    movie_index = new_df[new_df["title"]==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
inputMovie = input("Enter a movie: ")
recommend(inputMovie)
        
