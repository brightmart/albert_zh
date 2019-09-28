# albert_zh
海量中文语料上预训练ALBERT模型：参数更少，效果更好

Chinese version of ALBERT pre-trained model

*** UPDATE, 2019-09-28 ***  add code for three main changes of albert from bert and its test functions

ALBERT模型介绍
-----------------------------------------------
ALBERT模型是BERT的改进版，与最近其他State of the art的模型不同的是，这次是预训练小模型，效果更好、参数更少。

预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准

它对BERT进行了三个改造：

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

发布计划 Release Plan
-----------------------------------------------
1、albert_base, 参数量12M, 层数12，10月5号

2、albert_large, 参数量18M, 层数24，10月13号

3、albert_xlarge, 参数量59M, 层数24，10月6号

4、albert_xxlarge, 参数量233M, 层数12，10月7号（效果最佳的模型）

训练语料
-----------------------------------------------
40g中文语料，超过100亿汉字，包括多个百科、新闻、互动社区、小说、评论。

模型性能与对比(英文)
-----------------------------------------------    
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/state_of_the_art.jpeg"  width="80%" height="40%" />
  
   
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_performance.jpeg"  width="80%" height="40%" />

### 自然语言推断：XNLI

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


中文任务集上效果对比测试
-----------------------------------------------    

###  问题匹配语任务：LCQMC(Sentence Pair Matching)

| 模型 | 开发集(Dev) | 测试集(Test) |
| :------- | :---------: | :---------: |
| BERT | 89.4(88.4) | 86.9(86.4) | 
| ERNIE | 89.8 (89.6) | 87.2 (87.0) | 
| BERT-wwm |89.4 (89.2) | 87.0 (86.8) | 
| BERT-wwm-ext | - |-  |
| RoBERTa-zh-base | 88.7 | 87.0  |
| RoBERTa-zh-Large | 89.9(89.6) | 87.2(86.7) |
| RoBERTa-zh-Large(20w_steps) | 89.7| 87.0 |
| ALBERT-xlarge | ? | ? |
| ALBERT-xxlarge | ? | ? |


注：将很快替换？

模型参数和配置
-----------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/albert_configuration.jpeg"  width="80%" height="40%" />

代码实现和测试
-----------------------------------------------
通过运行以下命令测试主要的改进点，包括但不限于词嵌入向量参数的因式分解、跨层参数共享、段落连续性任务等。

    python test_changes.py


#### 技术交流与问题讨论QQ群: 836811304

If you have any question, you can raise an issue, or send me an email: brightmart@hotmail.com;

You can also send pull request to report you performance on your task or add methods on how to load models for PyTorch and so on.

If you have ideas for generate best performance pre-training Chinese model, please also let me know.

Reference
-----------------------------------------------
1、<a href="https://openreview.net/pdf?id=H1eA7AEtvS">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

2、<a href="http://baijiahao.baidu.com/s?id=1645712785366950083&wfr=spider&for=pc">预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准</a>

3、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

4、<a href="https://arxiv.org/abs/1907.10529">SpanBERT: Improving Pre-training by Representing and Predicting Spans</a>

5、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>




