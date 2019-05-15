## Links
Google Colab: https://colab.research.google.com/

ParlAI: https://github.com/facebookresearch/ParlAI

GPT-2 Chatbot: https://convai.huggingface.co/

My notes on 150 chatbot papers: https://github.com/ricsinaruto/Seq2seqChatbots/wiki


### Download ParlAI and install it inside Colaboratory
```
!git clone https://github.com/facebookresearch/ParlAI
%cd ParlAI
!python setup.py develop
```
### Start training a chatbot
```
!python examples/train_model.py -t dailydialog:noStart -bs 256 -m transformer/generator -mf train_files/DailyDialog/Transformer/big -eps 100 -sval true -veps 1 -vp 100 -vmt nll_loss -vmm min --shuffle true -ltim 10 -esz 512 -nl 6 -hid 512 --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --n-heads 8 -opt adam -clip 5 -tr 32 --dict-maxtokens 16384 -histsz 1 -lr 0.00002 -beta 0.9,0.98
```
### Chat with a trained chatbot
```
!python examples/interactive.py -mf train_files/DailyDialog/Transformer/big
```
