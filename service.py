
### imports

import os
from catboost import CatBoostClassifier
from fastapi import FastAPI, Query 
from typing import List
from schema import PostGet
from datetime import datetime
 

### loading model



def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/Users/daraluzina/ML/HW_22/Ver_2.0/catboost_model_4.cbm")  
    
    # Load CatBoost model
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    return model


model = load_models()

### loading features

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



def load_features() -> pd.DataFrame:
    df_users = batch_load_sql("SELECT * from public.user_data")
    df_posts = batch_load_sql("SELECT * from daria_luzina_features_post_3")
    return df_users, df_posts

df_users, df_posts = load_features()



### function to transform datasets to predictions and recommendations

def posts_recommendation(user_id, n=5):
    df = df_users[df_users['user_id']==user_id]
    df = df.merge(df_posts.drop(columns='text'), how='cross')
    df = df.drop(columns = 'user_id')
    df = df.set_index('post_id')
    df['predict_proba'] = model.predict_proba(df)[:,1]
    df = df.sort_values(by='predict_proba', ascending=False).head(n)
    df = df.merge(df_posts, left_index=True, right_on = 'post_id').reset_index()
    df = df.rename(columns={"post_id": "id"})
    df = df.rename(columns={"topic_x": "topic"})
    df = df[['id','text','topic']]
    list_of_dicts = df.to_dict('records')
    return list_of_dicts


### endpoint


app = FastAPI()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int,
    time: datetime = Query(...),  
    limit: int = Query(5)  
) -> List[PostGet]:
    return posts_recommendation(user_id=id, n=limit)
