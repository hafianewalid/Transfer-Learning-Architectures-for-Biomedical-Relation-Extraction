from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_transformers import BertModel as bm


def get_bert(bert_type='bert'):
    tokenizer, model = None, None
    if (bert_type == 'bert'):
        ######## bert ###########

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        ########################

    if (bert_type == 'biobert'):
        #### Bio BERT #########

        model = bm.from_pretrained('biobert_v1.1_pubmed')
        tokenizer = BertTokenizer(vocab_file="biobert_v1.1_pubmed/vocab.txt", do_lower_case=True)

        #### Bio BERT #########

    if (bert_type == 'scibert'):
        #### sci bert #########


        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=False)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)

        #######################

    return tokenizer, model
