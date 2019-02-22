import urllib.request
from pyquery import PyQuery as pq
from tqdm import tqdm
import pandas as pd	
import sqlite3
from sqlite3 import Error
from multiprocessing import Pool

NUM_PROCESS = 8
def get_page(url,data_id = None):
    try:
        f = urllib.request.urlopen(url)

        response = f.read().decode('utf-8')

        return response
    except Exception as e:
        if data_id is not None:
            print("An error occured when fetching the data ",data_id)
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
        return -1

def get_body(page_data,data_id = None):
    d = pq(page_data)
    content = d("#bodyContent")
    return content



 
def get_wiki_raw(x):
    data_id = x[0]
    print ("Processing ",data_id)
    url = x[1]
    response = get_page(url,data_id)
    body = "-1"
    if type(response) is str:
        body = get_body(response,data_id).html()
    return (data_id,url,body)

def save_wiki_raw(url_list,conn):
    url_list_temp = []

    for idx,x in tqdm(enumerate(url_list),ncols=100,total = len(url_list)):
        url_list_temp.append(x)
        if len(url_list_temp) == NUM_PROCESS or idx == len(url_list) - 1:
            print ("")
            #print (url_list_temp)
            with Pool(processes=NUM_PROCESS) as pool:
                url_list_result = pool.map(get_wiki_raw, url_list_temp)

            url_list_temp = []
            print ("updating")
            c = conn.cursor()
            for d in url_list_result:
            # Insert a row of data
                c.execute("insert into wiki_raw values (?,?,?)", d)

    # Save (commit) the changes
    conn.commit()
 
if __name__ == '__main__':
    conn = sqlite3.connect("../temp_result/wiki_url.db")

    gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv",sep='\t')
    gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv",sep='\t')
    gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv",sep='\t')

    train_url = list(zip(gap_train.ID,gap_train.URL))
    test_url = list(zip(gap_test.ID,gap_test.URL))
    valid_url = list(zip(gap_valid.ID,gap_valid.URL))

    #Create table
    c = conn.cursor()
    c.execute("create table if not exists wiki_raw (ID text,URL text,RAW_RESPONSE text)")



    save_wiki_raw(train_url,conn)
    save_wiki_raw(test_url,conn)
    save_wiki_raw(valid_url,conn)
    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()
    


