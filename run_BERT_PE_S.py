import application_bert_pe

ENCODER_ID = 'bert-base-uncased' # id from HuggingFace
MODEL_REFERENCE = 'BERT_PE_S'
N_EPOCHS = 4
LEARNING_RATE = 1e-5
MVPE_STEP = 1
#BATCH_SIZE = 8 # for GPUNodes
BATCH_SIZE = 16 # for RTX6000Node
DROPOUT_RATE = 0.1
USE_MLP = False
NORM_LAYER = False
EMBEDDING_DIM = 768
MAX_SENTENCE_LENGTH = 512

train_params = {}
train_params['mvpe_step'] = MVPE_STEP
train_params['learning_rate'] = LEARNING_RATE
train_params['n_epochs'] = N_EPOCHS
train_params['batch_size'] = BATCH_SIZE
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['use_mlp'] = USE_MLP
train_params['norm_layer'] = NORM_LAYER
train_params['max_sentence_len'] = MAX_SENTENCE_LENGTH
train_params['combination'] = 'sum'
train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8
train_params['use_mock'] = False
train_params['n_iterations'] = 5
#train_params['n_documents'] = 2 # for test. comment it when not in use

application_bert_pe.evaluate_BERT(train_params)
