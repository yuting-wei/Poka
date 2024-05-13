# Poka
The code of  the joint theme and emotion classification model of chinese classical poetry.

### Setup
+ Python >= 3.6
+ torch >= 0.4.1
+ numpy >= 1.17.4

### BERT-CCPoem Embedding
* Download Bert-CCPoem v1.0:

```
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/BERT_CCPoem_v1.zip
unzip BERT_CCPoem_v1.zip
```

### Train
```bash
python train.py config/poka_poetry.json
```

### Evaluation
```bash
python evaluate.py config/poka_poetry.json
```
