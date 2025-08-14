#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

books = pd.read_csv("./postprocessed_books.csv")


# In[ ]:


import tqdm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/sentence-t5-base')


# In[ ]:


books['description_embedding'] = books.apply(lambda row:  model.encode(row['description'] if len(row["description"]) > 0 else row["title"]).tolist(), axis=1)



# In[ ]:


books.to_csv('postprocessed_books_embeddings.csv', index=False)


# In[ ]:


books

