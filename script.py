#importing required libararies
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

## reading the given dataset
data = pd.read_excel("dux_network_data.xls")
print(data.head())

## removing not required columns from the dataset
not_requried = ["id", "VisitTime", "Profile", "Picture", "Degree", "Connections", "Summary", "Title", "From", "Company",
               "CompanyProfile", "CompanyWebsite", "PersonalWebsite", "Email", "Phone", "IM", "Twitter", "Location",
                "Industry","My Tags", "SalesProfile", "My Notes"]
data = data.drop(not_requried, axis=1)
print(data.head())

## making two dataframes for each highlighted sections (skills dataframe) (position based dataframe)
skill_Df = pd.DataFrame()
position_Df = pd.DataFrame()
for i, row in data.iterrows():
    indexes = list(row.index)
    name = row[0] + row[1] + row[2]
    pos = []
    skill = []
    for n in range(3,len(row)):
        if(pd.isna(row[n])):
            continue
        else:
            ind = indexes[n].split("-")
            if(ind[0] == "Skill"):
                skill.append(row[n])
                
            if(ind[0] == "Position" and ind[2] == "Description"):
                pos.append(row[n])
            else:
                continue
    if(len(skill)>1):
        skill_Df = skill_Df.append({"name": name, "skill": skill}, ignore_index=True)
    if(len(pos) > 1):
        position_Df = position_Df.append({"name": name,"position":pos}, ignore_index=True)

print(skill_Df.head())
print(position_Df.head())


## finding textual realationships between user using there skill set 
## creating word2vector model using skill sets of each user in the given dataset
model = Word2Vec(sentences=skill_Df.skill,min_count=1 ,vector_size=100,window=5, workers=4)
## function for checking similarity between the two users.
def get_edges_data(skill_df,i, model):
    sentence1 = skill_Df.skill[i]
    edges = []
    for n in range(0,len(skill_Df)):
        ## If users textual simialrity of skill set gets more then 94% then will add edge between the two users.
        if(model.wv.n_similarity(skill_Df.skill[i], skill_Df.skill[n]) > 0.94):
            if(i == n):
                continue
            else:
                edges.append((skill_Df.name[i], skill_Df.name[n]))
    return edges


##################################################################
## making graph of dataset using skills features from the dataset
##################################################################

## Intialization of the graph variable from networkx 
G = nx.Graph()
from tqdm import tqdm

## Loop over all of the rrow data in our dataset
for i in tqdm(range(0,len(skill_Df))):
    
    ## calling similiarity check function to find similarity between users
    edges = get_edges_data(skill_Df, i, model)
    
    ## If the node has correlated nodes with there similarity more then 94% then they will have edge
    if(len(edges)>0):
        for edge in edges:
            if(G.has_edge(edge[0], edge[1])):
                continue
            else:
                G.add_edge(edge[0], edge[1])
    else:
        ## This conditon will ignore those nodes which have zero connections with other nodes. 
        continue
        #G.add_node(skill_Df.name[i])

print(nx.info(G))

## simple python visualizaitons
# we have use the following function to create the visualizations of communities detected by the algorithm
import matplotlib.colors as mcolors
def draw_clu(G, pos, measures,name):
    clusters= list(set(measures.values()))
    plt.figure(figsize=(16,10))
    
    # Create the plot of the network to be placed in the figure
    nodes = nx.draw_networkx_nodes(G, pos, node_size=20, cmap=mcolors.ListedColormap(plt.cm.tab20(clusters)), 
                                   node_color=list(measures.values()),
                                   nodelist=list(measures.keys()))

    # Add edges to the plot
    edges = nx.draw_networkx_edges(G, pos,width=0.03, edge_color="r")
    labels = nx.draw_networkx_labels(G,pos, font_size=3, verticalalignment="top")
    plt.axis('off')
    plt.savefig("python visuals/"+name)
    plt.show()

Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
pos = nx.kamada_kawai_layout(G0)
import community
partitions = community.best_partition(G0)
draw_clu(G0, pos,partitions,"skillsgraph.pdf")

## saving skill set graph as gephi format for beautiful visualization
nx.write_gexf(G, "skillset.gexf")



##################################################################
## making graph of dataset using positon features from the dataset
##################################################################

stopwords=stopwords.words('english')
tokenizer = nltk.RegexpTokenizer(r"\w+") #for removel of punchuation 

## tokeingization and pre-processings to clean the textual features
def nltk_tokenizer(sentence):
    tokens = []
    for word in tokenizer.tokenize(sentence):
        word=word.lower()
        if word in stopwords:
            continue
        if (len(word) < 3): # words with length less then 1
            continue
        if (word == " "): #space
            continue
        if not (word.isalpha()):
            continue
        tokens.append(word)
    return tokens

ls = []
for text in (position_Df.position):
    st = " ".join(text)
    ls.append(st)


# creating pre-processed corpus of text
from tqdm import tqdm
processed_text = []

for text in tqdm(ls):
    tokens = ""
    for word in nltk_tokenizer(text.lower()):
        tokens+=word+" "
        
    processed_text.append(tokens)
print("The tokenized data is ready") 
position_Df["cleaned_text"] = processed_text
print(position_Df.head())

## finding relationship between users using there position features with the help of 
## TFIDF vectorizers and cosine similarities
def tf(data):
    print(">>TF-IDF partitioning process")
    tfidf = TfidfVectorizer(min_df = 5, max_df = 0.95, max_features = 10000,tokenizer=nltk_tokenizer).fit_transform(tqdm(data["cleaned_text"]))
    return tfidf
tfidf=tf(position_Df)
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf, tfidf)

dic = {}
i = 0
for sim in cosine_similarities:
    related_docs_indices = sim.argsort()[:-50:-1]
    ls  = []
    for indices in related_docs_indices:
        ls.append((indices, sim[indices]))
    dic[i] = ls
    i+=1

names = position_Df.name

## making second graph from the positon features
G2 = nx.Graph()
for i in range(0, len(position_Df)):
    for j in dic[i]:
        if(j[0] == i):continue
        elif(j[1]>0.25):
            if(G2.has_edge(names[i], names[j[0]])):continue
            else:G2.add_edge(names[i], names[j[0]])

print(nx.info(G2))
Gcc = sorted(nx.connected_components(G2), key=len, reverse=True)
G2 = G2.subgraph(Gcc[0])
pos = nx.kamada_kawai_layout(G2)

partition2 = community.best_partition(G2)
draw_clu(G2,pos,partition2, "positionsgraph.pdf")

