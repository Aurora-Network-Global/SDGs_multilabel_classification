# LOAD DATA

import pandas as pd
aurora_data=pd.read_hdf("C:/Datasets/aurora/50K_samples/SDGs_merged_cleaned_onehot_no_zeros_no_duplicates.h5")

# PREPARE DATA FOR mBERT

abstracts=aurora_data.Abstract.values
labels=aurora_data.iloc[:,4:].values

from transformers import BertConfig, BertTokenizer
from nltk import tokenize

tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
new_abstracts=[]

for abstract in abstracts:
    new_abstract="[CLS] "
    for sentence in tokenize.sent_tokenize(abstract):
        new_abstract=new_abstract + sentence + " [SEP] "
    new_abstracts.append(new_abstract)

abstracts=new_abstracts

MAX_LEN=400

tokenized_texts=[tokenizer.tokenize(sent)[:MAX_LEN] for sent in abstracts]

# use the BERT tokenizer to convert the tokens to their index numbers in the mBERT vocabulary
input_ids=[tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# pad input tokens
from keras.preprocessing.sequence import pad_sequences
input_ids=pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# divide data into: train, validation, test sets
train_inputs, temp_inputs, train_labels, temp_labels=train_test_split(input_ids, labels, random_state=1993, test_size=0.3)
validation_inputs, test_inputs, validation_labels, test_labels=train_test_split(temp_inputs, temp_labels, random_state=1993, test_size=0.5)

# create attention masks (of 1s for each token followed by 0s for padding)
train_masks=[]

for seq in train_inputs:
    seq_mask=[float(i>0) for i in seq]
    train_masks.append(seq_mask)


validation_masks=[]

for seq in validation_inputs:
    seq_mask=[float(i>0) for i in seq]
    validation_masks.append(seq_mask)


test_masks=[]

for seq in test_inputs:
    seq_mask=[float(i>0) for i in seq]
    test_masks.append(seq_mask)

# convert everything to tensors
from tensorflow import convert_to_tensor

train_inputs=convert_to_tensor(train_inputs)
validation_inputs=convert_to_tensor(validation_inputs)
test_inputs=convert_to_tensor(test_inputs)

train_labels=convert_to_tensor(train_labels)
validation_labels=convert_to_tensor(validation_labels)
test_labels=convert_to_tensor(test_labels)

train_masks=convert_to_tensor(train_masks) 
validation_masks=convert_to_tensor(validation_masks)
test_masks=convert_to_tensor(test_masks)

# COUNT WEIGHTS

def count_labels(dataframe):
    output=pd.DataFrame(columns=["label", "count"])
    labels=dataframe.columns[4:]
    for label in labels:
        output=output.append({"label": label,
                                "count": sum(dataframe[label])}, ignore_index=True)
    return output

labels_counts=count_labels(aurora_data)

# see the distribution of targets
import plotly.express as px

fig=px.bar(labels_counts, x="label", y="count", title="Distribution of targets in Aurora 50k samples dataset")

fig.show()

labels_counts["weight"]=len(labels_counts)/labels_counts["count"]

class_weights=pd.Series(labels_counts.weight.values, index=[_ for _ in range(170)]).to_dict()

# CREATE MODEL

from transformers import TFBertModel, BertConfig

config=BertConfig.from_pretrained("bert-base-multilingual-uncased", num_labels=170)
bert=TFBertModel.from_pretrained("bert-base-multilingual-uncased", config=config)
bert_layer=bert.layers[0]

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model

input_ids_layer=Input(shape=(MAX_LEN),
                        name="input_ids",
                        dtype="int32")

input_attention_masks_layer=Input(shape=(MAX_LEN),
                                    name="attention_masks",
                                    dtype="int32")

bert_model=bert_layer(input_ids_layer, input_attention_masks_layer)

target_layer=Dense(units=170,
                    kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                    name="target_layer",
                    activation="sigmoid")(bert_model[1])

model=Model(inputs=[input_ids_layer, input_attention_masks_layer],
            outputs=target_layer)

from tensorflow.keras.optimizers import Adam

optimizer=Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

from keras.metrics import Precision, Recall

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy", 
    metrics=[Precision(), Recall()])

# TRAIN MODEL

history=model.fit([train_inputs, train_masks], train_labels,
    batch_size=2,
    epochs=3,
    class_weight=class_weights,
    validation_data=([validation_inputs, validation_masks], validation_labels))

# TEST MODEL

test_score=model.evaluate([test_inputs, test_masks], test_labels,
                batch_size=2)

