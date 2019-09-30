# albert_zh

An Implementation of <a href="https://arxiv.org/pdf/1909.11942.pdf">A Lite Bert For Self-Supervised Learning Language Representations</a> with TensorFlow

海量中文语料上预训练ALBERT模型：参数更少，效果更好。预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准

Chinese version of ALBERT pre-trained model, both TensorFlow and PyTorch checkpoint of Chinese will be available 

*** UPDATE, 2019-10-01 ***  

     Relesed albert_base_zh with only 10% parameters of bert_base, very small model(40M) & training can be very fast. 

*** UPDATE, 2019-09-28 ***  add code for three main changes of albert from bert and its test functions

模型下载
-----------------------------------------------
1、<a href="https://storage.googleapis.com/albert_zh/albert_base_zh.zip">albert_base_zh(小模型体验版)</a>, 参数量12M, 层数12，大小为40M

    参数量为bert_base的十分之一，模型大小也十分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base下降约1个点；相比未预训练，albert_base提升14个点

2、albert_large, albert_xlarge, albert_xxlarge, coming soon.

    if you want albert model with best performance, there is still a few days to go.

ALBERT模型介绍 Introduction of ALBERT
-----------------------------------------------
ALBert is based on Bert, but with some improvements. It achieve state of the art performance on main benchmarks recently, but with

30% parameters less or more.

ALBERT模型是BERT的改进版，与最近其他State of the art的模型不同的是，这次是预训练小模型，效果更好、参数更少。

它对BERT进行了三个改造 Three main changes of ALBert from Bert：

1）词嵌入向量参数的因式分解 Factorized embedding parameterization
   
     O(V * H) to O(V * E + E * H)
     
     如以ALBert_xxlarge为例，V=30000, H=4096, E=128
       
     那么原先参数为V * H= 30000 * 4096 = 1.23亿个参数，现在则为V * E + E * H = 30000*128+128*4096 = 384万 + 52万 = 436万，
       
     词嵌入相关的参数变化前是变换后的28倍。


2）跨层参数共享 Cross-Layer Parameter Sharing

     参数共享能显著减少参数。共享可以分为全连接层、注意力层的参数共享；注意力层的参数对效果的减弱影响小一点。

3）段落连续性任务 Inter-sentence coherence loss.
     
     使用段落连续性任务。正例，使用从一个文档中连续的两个文本段落；负例，使用从一个文档中连续的两个文本段落，但位置调换了。
     
     避免使用原有的NSP任务，原有的任务包含隐含了预测主题这类过于简单的任务。

      We maintain that inter-sentence modeling is an important aspect of language understanding, but we propose a loss 
      based primarily on coherence. That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which avoids topic 
      prediction and instead focuses on modeling inter-sentence coherence. The SOP loss uses as positive examples the 
      same technique as BERT (two consecutive segments from the same document), and as negative examples the same two 
      consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about
      discourse-level coherence properties. 

其他变化，还有 Other changes：

    1）去掉了dropout  Remvoe dropout to enlarge capacity of model.
        最大的模型，训练了1百万步后，还是没有过拟合训练数据。说明模型的容量还可以更大，就移除了dropout（dropout可以认为是随机的去掉网络中的一部分，同时使网络变小一些）
        We also note that, even after training for 1M steps, our largest models still do not overfit to their training data. As a result, we decide to remove dropout to further increase our model capacity.
        其他型号的模型，在我们的实现中我们还是会保留原始的dropout的比例，防止模型对训练数据的过拟合。
        
    2）为加快训练速度，使用LAMB做为优化器 Use lAMB as optimizer, to train with big batch size
      使用了大的batch_size来训练(4096)。 LAMB优化器使得我们可以训练，特别大的批次batch_size，如高达6万。
    
    3）使用n-gram(uni-gram,bi-gram, tri-gram）来做遮蔽语言模型 Use n-gram as make language model
       即以不同的概率使用n-gram,uni-gram的概率最大，bi-gram其次，tri-gram概率最小。
       本项目中目前使用的是在中文上做whole word mask，稍后会更新一下与n-gram mask的效果对比。n-gram从spanBERT中来。

发布计划 Release Plan
-----------------------------------------------
1、albert_base, 参数量12M, 层数12，10月7号

2、albert_large, 参数量18M, 层数24，10月13号

3、albert_xlarge, 参数量59M, 层数24，10月6号

4、albert_xxlarge, 参数量233M, 层数12，10月7号（效果最佳的模型）

训练语料/训练配置 Training Data & Configuration
-----------------------------------------------
30g中文语料，超过100亿汉字，包括多个百科、新闻、互动社区。

预训练序列长度sequence_length设置为512，批次batch_size为4096，训练产生了3.5亿个训练数据(instance)；每一个模型默认会训练125k步，albert_xxlarge将训练更久。

作为比较，roberta_zh预训练产生了2.5亿个训练数据、序列长度为256。由于albert_zh预训练生成的训练数据更多、使用的序列长度更长，
 
    我们预计albert_zh会有比roberta_zh更好的性能表现，并且能更好处理较长的文本。

训练使用TPU v3 Pod，我们使用的是v3-256，它包含32个v3-8。每个v3-8机器，含有128G的显存。



模型性能与对比(英文) Performance and Comparision
-----------------------------------------------    
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/state_of_the_art.jpg"  width="80%" height="40%" />
  
   
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_performance.jpg"  width="80%" height="40%" />


<img src="https://github.com/brightmart/albert_zh/blob/master/resources/add_data_removing_dropout.jpg"  width="80%" height="40%" />


中文任务集上效果对比测试 Performance on Chinese datasets
----------------------------------------------- 

### 自然语言推断：XNLI of Chinese Version

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) | 
| ERNIE | 79.7 (79.4) | 78.6 (78.2) | 
| BERT-wwm | 79.0 (78.4) | 78.2 (78.0) | 
| BERT-wwm-ext | 79.4 (78.6) | 78.7 (78.3) |
| XLNet | 79.2  | 78.7 |
| RoBERTa-zh-base | 79.8 |78.8  |
| RoBERTa-zh-Large | 80.2 (80.0) | 79.9 (79.5) |
| ALBERT-xlarge | ? | ? |
| ALBERT-xxlarge | ? | ? |

注：BERT-wwm-ext来自于<a href="https://github.com/ymcui/Chinese-BERT-wwm">这里</a>；XLNet来自于<a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">这里</a>; RoBERTa-zh-base，指12层RoBERTa中文模型
   
###  问题匹配语任务：LCQMC(Sentence Pair Matching)

| 模型 | 开发集(Dev) | 测试集(Test) |
| :------- | :---------: | :---------: |
| BERT | 89.4(88.4) | 86.9(86.4) | 
| ERNIE | 89.8 (89.6) | 87.2 (87.0) | 
| BERT-wwm |89.4 (89.2) | 87.0 (86.8) | 
| BERT-wwm-ext | - |-  |
| RoBERTa-zh-base | 88.7 | 87.0  |
| RoBERTa-zh-Large | ***89.9(89.6)*** | ***87.2(86.7)*** |
| RoBERTa-zh-Large(20w_steps) | 89.7| 87.0 |
| ALBERT-zh-base | 86.4 | 86.3 |
| ALBERT-xlarge | ? | ? |
| ALBERT-xxlarge | ? | ? |

### 

| Model | MLM eval acc | SOP eval acc | Training(Hours) | Loss eval |
| :------- | :---------: | :---------: | :---------: |:---------: |
| albert_zh_base | 79.1% | 99.0% | 6h | 1.01|
| albert_zh_large | ? | ? | ? | ?|


注：? 将很快替换

模型参数和配置 Configuration of Models
-----------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_configuration.jpg"  width="80%" height="40%" />

代码实现和测试 Implementation and Code Testing
-----------------------------------------------
通过运行以下命令测试主要的改进点，包括但不限于词嵌入向量参数的因式分解、跨层参数共享、段落连续性任务等。

    python test_changes.py

预训练 Pre-training
-----------------------------------------------

#### 生成特定格式的文件(tfrecords) Generate tfrecords Files

运行以下命令即可。项目自动了一个示例的文本文件(data/news_zh_1.txt)
   
       bash create_pretrain_data.sh
   
如果你有很多文本文件，可以通过传入参数的方式，生成多个特定格式的文件(tfrecords）

#### 执行预训练 pre-training on GPU/TPU
    GPU:
    export BERT_BASE_DIR=albert_config
    nohup python3 run_pretraining.py --input_file=./data/tf*.tfrecord  \
    --output_dir=my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/albert_config_xxlarge.json \
    --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=76 \
    --num_train_steps=125000 --num_warmup_steps=12500 --learning_rate=0.00176    \
    --save_checkpoints_steps=2000   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt &
    
    TPU, add following information:
        --use_tpu=True  --tpu_name=grpc://10.240.1.66:8470 --tpu_zone=us-central1-a
        
    注：如果你重头开始训练，可以不指定init_checkpoint；
    如果你从现有的模型基础上训练，指定一下BERT_BASE_DIR的路径，并确保bert_config_file和init_checkpoint两个参数的值能对应到相应的文件上；
    领域上的预训练，根据数据的大小，可以不用训练特别久。


#### 技术交流与问题讨论QQ群: 836811304 Join us on QQ group

If you have any question, you can raise an issue, or send me an email: brightmart@hotmail.com;

You can also send pull request to report you performance on your task or add methods on how to load models for PyTorch and so on.

If you have ideas for generate best performance pre-training Chinese model, please also let me know.

##### Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)

Reference
-----------------------------------------------
1、<a href="https://openreview.net/pdf?id=H1eA7AEtvS">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

2、<a href="http://baijiahao.baidu.com/s?id=1645712785366950083&wfr=spider&for=pc">预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准</a>

3、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

4、<a href="https://arxiv.org/abs/1907.10529">SpanBERT: Improving Pre-training by Representing and Predicting Spans</a>

5、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>




