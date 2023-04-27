from docarray import Document, DocumentArray
from tqdm import tqdm
from text2vec import SentenceModel, EncoderType
import numpy as np
import os,torch
from pprint import pprint
os.environ['DEVICE'] = 'cuda'

with open('./三体I：地球往事.txt',encoding='GBK') as f:
    txt = f.read()
# d = Document(
#     uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()  # 链接是傲慢与偏见的电子书，此处将电子书内容加载到 Document 中
d = Document(text = txt)
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())  # 按照换行进行分割字符串

# da.apply(lambda d: d.embed_feature_hashing())#仅仅是个词袋模型，可改进

model = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG, device='cuda')
feature_vec = model.encode
print('model done!')

for d in tqdm(da):
    d.embedding = feature_vec(d.text)

text = Document(text = '警告地球人不要回答')
text.embedding = feature_vec(text.text)
q = text.match(da, limit=10, exclude_self=True, metric='cos', use_scipy=True)
pprint(q.matches[:,('text','scores__cos')])
# q = (
#     Document(text='she entered the room')  # 要匹配的文本
#     .embed_feature_hashing()  # 通过 hash 方法进行特征编码
#     .match(da, limit=5, exclude_self=True, metric='jaccard', use_scipy=True)  # 找到五个与输入的文本最相似的句子
# )
#
# print(q.matches[:, ('text', 'scores__jaccard')])  # 输出对应的文本与 jaccard 相似性分数