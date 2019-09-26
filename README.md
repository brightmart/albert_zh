# albert_zh
海量中文语料上预训练ALBERT模型：参数更少，效果更好

Chinese version of ALBERT pre-trained model

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

2、albert_large, 参数量18M, 层数24，10月5号

3、albert_xlarge, 参数量59M, 层数24，10月6号（体验版）

4、albert_xxlarge, 参数量233M, 层数12，10月13号

训练语料
-----------------------------------------------
40g中文语料，超过100亿汉字，包括多个百科、新闻、互动社区、小说、评论。

模型性能与对比
----------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/Ralbert_configuration.jpeg"  width="70%" height="60%" />


模型参数和配置
-----------------------------------------------
<img src="https://github.com/brightmart/albert_zh/blob/master/resources/Ralbert_configuration.jpeg"  width="70%" height="60%" />

Reference
-----------------------------------------------
<a href="https://openreview.net/pdf?id=H1eA7AEtvS">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

<a href="http://baijiahao.baidu.com/s?id=1645712785366950083&wfr=spider&for=pc">预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准</a>

<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

<a href="https://arxiv.org/abs/1907.10529">SpanBERT: Improving Pre-training by Representing and Predicting Spans</a>



