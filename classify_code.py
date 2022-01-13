##### IMPORTED MODULES ######
import numpy
import sklearn
import math
import nltk
import time
import numpy as np 
import json
from nltk.stem import PorterStemmer
from os.path import exists as file_exists
import os 
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')



closed_class_categories = ['CD','CC','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','RP','TO','UH','WDT','WP','WP$','WRB']


def prettify(element,indent= ''):
    queue = [(0,element)]
    while queue:
        level,element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = '\n' + indent * (level+1)
        if queue:
            element.tail = '\n' + indent * queue[0][0]
        else:
            element.tail = '\n' + indent * (level-1)
        queue[0:0] = children




#%%
###### READ FILES ######


### TRAIN ###      

articles_n_vocabulary_train = {}

with os.scandir('C:/Users/odavr/OneDrive/Υπολογιστής/classify_articles/20news-bydate-train') as train_folder:
    for line in train_folder:
        whole_text = ''
        with os.scandir(line) as topic:
            txt_count=0
            for txt_file in topic:
                txt_count+=1
                with open(txt_file,'r',encoding="utf-8",errors='ignore') as text:
                    for str_text in text:
                        whole_text += str_text
                if txt_count == 100: #taking 200 txts from each category
                    break
        articles_n_vocabulary_train[line.name] = whole_text 
   
        

            


#%%

###### RENAMING TOPICS IN TO A CLEANER TITLE ######

topics = ['atheism','graphics','windows','ibm','mac','windows_2','sales','cars','motorcycle','baseball','hockey','cryptography','electronics','medics','space','christianism',
'guns','mideast','politics','religion']


    
new_name_train = 0
for old_name_train in articles_n_vocabulary_train.copy():
    articles_n_vocabulary_train[topics[new_name_train]] =  articles_n_vocabulary_train.pop(old_name_train)
    new_name_train +=1


#%%


###### CLEANING THE VOCABULARY OF EACH DATASET ######

inverted_file_train = {} #inverted file for topics
num_of_words_in_top_train={} # number of words in each topic


for turn in range(len(articles_n_vocabulary_train)):
    
    
    print('{} out of {}'.format(turn+1,len(articles_n_vocabulary_train)))
    
    sentenses = nltk.sent_tokenize(articles_n_vocabulary_train[topics[turn]])
    tokenized_words=[]
    for sent in sentenses:
        tokenized_words += nltk.word_tokenize(sent)
        
          
    #remove stopwords
    no_stop_words_list = []
    num_of_words = 0
    
    
    for w in tokenized_words:
        
        
        if w in stopwords:
            
            continue
        else:
            num_of_words+=1
            no_stop_words_list.append(w)
    
    num_of_words_in_top_train[topics[turn]]=num_of_words #save the number of words, after excluding stopwords, for each article
      
    ### remove punctuation
    symbols = "!\"#$%&()*+-,./:;<=>?@[\]^_`{|}~\n"
    for symbol in symbols:
        no_stop_words_list = np.char.replace(no_stop_words_list,symbol,'')
    
    ### remove apostrophe
    no_stop_words_list = np.char.replace(no_stop_words_list, "'", " ")
       
    ###remove single characters

    no_stop_words_list_completed = []
    for i in no_stop_words_list:
        one_part = i.split(' ')
        for j in one_part:
            if j not in symbols:
                no_stop_words_list_completed.append(j)
     
    ###remove single characters
            
    a_list = no_stop_words_list_completed
    for www in range(len(no_stop_words_list_completed)):
        if len(no_stop_words_list_completed[www]) > 1:
            a_list.append(no_stop_words_list_completed[www])
    no_stop_words_list_completed = a_list
     
    no_numbers_words_train = [item for item in no_stop_words_list_completed if item.isalpha()]       
       
    clean_words_train=[]
    for i in range(len(no_numbers_words_train)):
        if no_numbers_words_train[i].isalnum():
            clean_words_train.append(no_numbers_words_train[i])
       
    #stemming words
    porter = PorterStemmer()
    stemmed_words_train = []
    for jjword in no_stop_words_list_completed:
        jjword = jjword.lower()
        stem = porter.stem(jjword)
        stemmed_words_train.append(stem)
       
    sorted_list_train = []
    for i in stemmed_words_train:    
        sorted_list_train.append(i)
    sorted_list_train = sorted(sorted_list_train)
    
        
       
     #inverted file creation
    for word in stemmed_words_train:
        if word not in inverted_file_train:
            inverted_file_train[word] = tuple([[turn,1]])
        else:
            exists = 0
            Tuple_to_List = list(inverted_file_train[word])
            for j in range(len(Tuple_to_List)):
                if Tuple_to_List[j][0] == turn:
                    Tuple_to_List[j][1] +=1
                    exists = 1 
            if exists != 1:
                Tuple_to_List.append([turn,1])

            inverted_file_train[word] = tuple(Tuple_to_List)
      
# saving inverted file for 20newsgroup

if file_exists('20newsgroups_inverted_file_train.json')==False or os.path.getsize('20newsgroups_inverted_file_train.json') == 0:
    with open('20newsgroups_inverted_file_train.json','a+') as file:
        file.write(json.dumps(inverted_file_train))
        file.close()
else:
    with open('20newsgroups_inverted_file_train.json','r') as file:
        file_data = json.load(file)
        file_data.update(inverted_file_train)
        file.close()
        
#saving the dictionary with pairs of 1)topics and 2)the number of words that contain respectivly

if file_exists('Number_of_words_in_topics_train.json')==False or os.path.getsize('Number_of_words_in_topics_train.json') == 0:
    with open('Number_of_words_in_topics_train.json','a+') as file:
        file.write(json.dumps(num_of_words_in_top_train))
        file.close()
else:
    with open('Number_of_words_in_topics_train.json','r') as file:
        file_data = json.load(file)
        file_data.update(num_of_words_in_top_train)
        file.close()
#%% 

###LOADING DATA FOR INVERTED FILE TRAIN AND NUMBER OF WORDS IN TOPICS TRAIN

with open('20newsgroups_inverted_file_train.json','r') as file:
    inverted_file_train = json.load(file)

with open('Number_of_words_in_topics_train.json','r') as file:
    num_of_words_in_top_train = json.load(file)





#%%

### CREATING INVERTED INDEX XML FILE FOR TRAIN DATA SET

import xml.etree.ElementTree as ET


count= 0
start = time.time()
xml_doc = ET.Element('twenty_newsgroup_inverted_index')

    
for word in inverted_file_train:
    
    count+=1
    
    print('{} OUT OF {}'.format(count,len(inverted_file_train)))
    
    if word.isalnum() == False:
        continue
    
    
    lemmas = ET.SubElement(xml_doc,'lemma',name=word)
    
    total_words_in_article = 0
    for cur_id in range(len(topics)):
        
        #how many times does the word appear in this topic
        appears_in_the_doc = 0
        
        for term in range(len(inverted_file_train[word])):
            
            
            if inverted_file_train[word][term][0]==cur_id:
                appears_in_the_doc = inverted_file_train[word][term][1]
                break

        #total words in the document
        
        total_words_in_article = num_of_words_in_top_train[topics[cur_id]]
                    
        
        tf = appears_in_the_doc / total_words_in_article
        
        
        #idf = log(total number of documents/number of documents where the term appears)
        idf = math.log(len(topics)/len(inverted_file_train[word]))
        
        
        tf_idf = tf*idf
        if(tf_idf) != 0:
            ET.SubElement(lemmas,'document id = "{}" TF-IDF = "{}"'.format(cur_id,tf_idf))
   

prettify(xml_doc) 
tree = ET.ElementTree(xml_doc)
print('Data are being saved ...')
tree.write('20newsgroup_Inverted_Index_Train.xml',encoding='utf-8',xml_declaration=True)
end = time.time() 

print('Total time runned: {}'.format(end-start))

#%% 

"""
READ 20newsgroups_inverted_index to extract the first 300 highest
tf-idf weights for each topic in order to create the vector
"""


import xml.etree.ElementTree as ET
mytree=ET.parse('20newsgroup_Inverted_Index_Train.xml')
myroot = mytree.getroot()

highest_200_tf_idf = {}

for lemma in myroot.findall('lemma'):
    
    for doc in lemma:
        if doc.attrib['id'] in highest_200_tf_idf:
            highest_200_tf_idf[doc.attrib['id']] += [[lemma.attrib['name'],doc.attrib['TF-IDF']]]
        else:
            highest_200_tf_idf[doc.attrib['id']] = [[lemma.attrib['name'],doc.attrib['TF-IDF']]]
    


words_and_tf_idf = {}

for capa in highest_200_tf_idf:   #topic's level
    words_for_vector=[]
    list_to_sort = []
    for i in highest_200_tf_idf[capa]:  # words in topics level
        list_to_sort.append(i[1])       #store tf-idfs' weight
        
    tf_idf_sorted = sorted(list_to_sort)     #sort that list
    
    top_values = tf_idf_sorted[-1:len(tf_idf_sorted)-300-1:-1]   #take the top values 
    #precentage_ignored_20 = len(highest_200_tf_idf[capa]) * 0.20
    #top_values = tf_idf_sorted[-int(precentage_ignored_20):len(tf_idf_sorted)-int(precentage_ignored_20)-400-1:-1]    # i ignore 25% from top weights
    
    
    #process to find the word representing each value
    for big_val in top_values:                         #for every stored value          
        
        for j in range(len(highest_200_tf_idf[capa])):     #for every pair word-weight
            if highest_200_tf_idf[capa][j][1]==big_val:    
                words_for_vector.append(highest_200_tf_idf[capa][j][0])
                print(highest_200_tf_idf[capa].pop(j))
                break
                
                
   
    words_and_tf_idf[capa] = words_for_vector
#%%
#### CREATING THE VECTOR FOR COMPARISON ####
  
#Xwros xarakthristikwn S = 8000
vector = []
for i in words_and_tf_idf:
    for j in words_and_tf_idf[i]:
        vector.append(j)
print(len(vector))

#%%

##### VECTORS FOR EACH TOPIC ####

vectors_n_topics = {}

with open('Number_of_words_in_topics_train.json') as f:
    topic_size = json.load(f)


for topicz in words_and_tf_idf:
    
    vector_list = []
    for v_word in vector:
        for li in range(len(inverted_file_train[v_word])):
            if inverted_file_train[v_word][li][0] == int(topicz):
                occurancies = inverted_file_train[v_word][li][1]
                tf = 1 + math.log(occurancies/(topic_size[topics[int(topicz)]]))
                break 
            else:
                tf=0
        
        idf = math.log(1+20/(len(inverted_file_train[i])))
        tfidf_value=tf*idf
        
        vector_list.append(tfidf_value)
    vectors_n_topics[topicz] = vector_list




#%% 

#### TRYING TO FIND THE VECTORS FOR EACH TEST ARTICLE ####

### Im gonna take 30 articles from each category ###


articles_n_vocabulary_test = {} # here storing like {category: ['txt1','txt2',...]}

with os.scandir('C:/Users/odavr/OneDrive/Υπολογιστής/classify_articles/20news-bydate-test') as test_folder:
    for line in test_folder:
        
        with os.scandir(line) as topic:
            txt_count=0
            for txt_file in topic:
                txt_count+=1
                whole_text = ''
                with open(txt_file,'r',encoding="utf-8",errors='ignore') as text:
                    for str_text in text:
                        whole_text +=str_text
                try:
                    articles_n_vocabulary_test[line.name].append(whole_text)
                except:
                    articles_n_vocabulary_test[line.name] = [whole_text]
            
                if txt_count == 7:
                    break

new_name_test = 0
for old_name_test in articles_n_vocabulary_test.copy():
    articles_n_vocabulary_test[topics[new_name_test]] = articles_n_vocabulary_test.pop(old_name_test)
    new_name_test +=1
            
 #%%

#### Cleaning the articles ####

vectors_n_articles ={}

num_of_words_in_top_test = {}


for ttle in articles_n_vocabulary_test:
    for sub_list in range(len(articles_n_vocabulary_test[ttle])):
        
        sentences = nltk.sent_tokenize(articles_n_vocabulary_test[ttle][sub_list])
        tokenized_words =[]
        for sent in sentences:
            tokenized_words += nltk.word_tokenize(sent)

    #remove stopwords
    
    
        no_stop_words_list = []
        
        for w in tokenized_words:
            if w in stopwords:
                 continue
            else:
                no_stop_words_list.append(w)
         
    
    
    
        symbols = "!\"#$%&()*+-,./:;<=>?@[\]^_`{|}~\n"
        for symbol in symbols:
            no_stop_words_list = np.char.replace(no_stop_words_list,symbol,' ')
        
        ### remove apostrophe
        no_stop_words_list = np.char.replace(no_stop_words_list, "'", "")
        
        
        no_stop_words_list_completed = []
        for i in no_stop_words_list:
            one_part = i.split(' ')
            for j in one_part:
                if j not in symbols:
                    no_stop_words_list_completed.append(j)
    
        ###remove single characters
                
        a_list = no_stop_words_list_completed
        for www in range(len(no_stop_words_list_completed)):
            if len(no_stop_words_list_completed[www]) > 1:
                a_list.append(no_stop_words_list_completed[www])
        no_stop_words_list_completed = a_list
                
        no_numbers_words = [item for item in no_stop_words_list_completed if item.isalpha()]
        
        clean_words=[]
        for i in range(len(no_numbers_words)):
            if no_numbers_words[i].isalnum():
                clean_words.append(no_numbers_words[i])
        
        porter = PorterStemmer()
        stemmed_words = []
        for jjword in clean_words:
            jjword = jjword.lower()
            stem = porter.stem(jjword)
            stemmed_words.append(stem)
            
        sorted_list = []
        for i in stemmed_words:    
            sorted_list.append(i)
        sorted_list = sorted(sorted_list)
    

        final_stage_list,frequency = np.unique(sorted_list, return_counts=True)
        words_n_appearances = np.array(final_stage_list)
        words_n_appearances = list(zip(words_n_appearances, frequency))
       
        ############## Vector for article ###########
        
        
    
        vector_list = []
        for i in vector:
            
            if i in final_stage_list:
                for pair in words_n_appearances:
                    if pair[0] == i:
                        times_appear = pair[1]
                tf = 1 + math.log(times_appear/(len(no_numbers_words))) # number of times word appears in article
                idf = math.log(1+20/(len(inverted_file_train[i])))
                tfidf_value = tf*idf
            else:
                tfidf_value = 0
            vector_list.append(tfidf_value)
    
        try:
            vectors_n_articles[ttle].append(vector_list)
        except:
            vectors_n_articles[ttle] = [vector_list]
        print('{} vector done'.format(sub_list+1))
    print("{} DONE".format(ttle))  
    
#%%

from scipy.spatial import distance

hand_categorized_articles = {}

for title in vectors_n_articles: # from every topic
    count_vec = 0 #count how many vectors has been classified
    for vec in vectors_n_articles[title]: #we take a vector
        count_vec+=1
        last_cosine_similarity = 0
        for topic in range(20):  # and it is compared with every other already classified vector 
            vec_1 = vec
            vec_2 = vectors_n_topics['{}'.format(topic)]
            
            cosine_similarity = 1 - distance.cosine(vec_1, vec_2)
            
            #print("{}, {}: {:.20f} ".format(title,topic,cosine_similarity))
            if cosine_similarity >= last_cosine_similarity:
                hand_categorized_articles.update({'{},{}'.format(title,count_vec):topics[int(topic)]})
                last_cosine_similarity = cosine_similarity
            else:
                continue


        
#%%

for result in hand_categorized_articles:
    success = 
    if hand_categorized_articles[rusult] == result:








