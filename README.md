Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

Teacher Model
=============
Download Hugging Face bert-base-uncased model as teacher to **BERT_BASE_DIR**

```
git lfs install
git clone https://huggingface.co/bert-base-uncased
```

Corpus
=======
Raw english wikipedia and bookcorpus dataset can be downloaded via this [link](https://drive.google.com/drive/folders/1kzQKL9LQxgmsOBlKWVNI_gixY4uYdPGl?usp=sharing).




General Distillation
====================

General distillation has two steps: (1) generate the corpus of json format; (2) run the transformer distillation;

Step 1: use `pregenerate_training_data.py` to produce the corpus of json format  


```
 
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: use `general_distill.py` to run the general distillation
```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```







