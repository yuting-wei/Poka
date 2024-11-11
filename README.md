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

## Citation

Please cite our paper if you use our code, dataset, or compare with our model:
```
@ARTICLE{10737425,
  author={Wei, Yuting and Hu, Linmei and Zhu, Yangfu and Zhao, Jiaqi and Wu, Bin},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Knowledge-Guided Transformer for Joint Theme and Emotion Classification of Chinese Classical Poetry}, 
  year={2024},
  volume={32},
  pages={4783-4794},
  doi={10.1109/TASLP.2024.3487409}}
```
