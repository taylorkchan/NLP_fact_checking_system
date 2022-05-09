from openie import StanfordOpenIE
import string
import spacy
from itertools import combinations
import os, sys

import numpy as np
import transformers
from termcolor import colored

# load NeuralCoref and add it to the pipe of SpaCy's model
import neuralcoref


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import tensorflow as tf
print(tf.__version__)

nlp = spacy.load('en_core_web_lg')

coref = neuralcoref.NeuralCoref(nlp.vocab, conv_dict={'Singapore': ['country', 'region', 'area']})
nlp.add_pipe(coref, name='neuralcoref')

with HiddenPrints():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=batch_size,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            max_length=max_length,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


def initialize_model():

    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_outputs = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bert_outputs = bert_outputs[0]
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(bert_outputs)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


    # Create a new model instance
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    # Restore the weights
    model.load_weights('./training_1/cp.ckpt')
    model.save('model_dir')
    return model

model = initialize_model()

def process_claim(claim_statement):
    doc = nlp(claim_statement)
    doc1 = nlp(doc._.coref_resolved)
    claims = []
    client = StanfordOpenIE()
    text = doc1.text
    print('Text: %s' % text)
    # print(client.annotate(text))
    for triple in client.annotate(text):
        # print(triple)
        # print(triple["subject"])
        # print(triple["relation"])
        # print(triple["object"])
        claims.append([triple["subject"], triple["relation"], triple["object"]])
    if (len(claims) == 0):
        print('split and try again')
        claim_statements = [claim_statement[i: i + 100] for i in range(0, len(claim_statement), 100)]
        for claim_stmt in claim_statements:
            doc = nlp(claim_statement)
            doc1 = nlp(doc._.coref_resolved)
            sents = list(doc1.sents)
            for triple in client.annotate(text):
                # print(triple)
                # print(triple["subject"])
                # print(triple["relation"])
                # print(triple["object"])
                claims.append([triple["subject"], triple["relation"], triple["object"]])
    claim_statement_no_punc = claim_statement.translate(claim_statement.maketrans('', '', string.punctuation))
    doc = nlp(claim_statement_no_punc)
    doc1 = nlp(doc._.coref_resolved)
    text = doc1.text
    annotations = client.annotate(text)
    for triple in annotations:
        claims.append([triple["subject"], triple["relation"], triple["object"]])
    if (len(annotations) == 0):
        print('split and try again')
        claim_statements = [claim_statement[i: i + 100] for i in range(0, len(claim_statement), 100)]
        for claim_stmt in claim_statements:
            doc = nlp(claim_statement)
            doc1 = nlp(doc._.coref_resolved)
            sents = list(doc1.sents)
            for triple in client.annotate(text):
                # print(triple)
                # print(triple["subject"])
                # print(triple["relation"])
                # print(triple["object"])
                claims.append([triple["subject"], triple["relation"], triple["object"]])

    claims_dedup = []
    for i in claims:
        if i not in claims_dedup:
            print(i)
            claims_dedup.append(i)
    if len(claims_dedup)>8:
        for i in claims_dedup:
            i.append(len(" ".join(i)))
        from operator import itemgetter
        claims_dedup = sorted(claims_dedup, key=itemgetter(3))[-8:]
        print("kept 8 only")
        for i in claims_dedup:
            i.pop()
        print(claims_dedup)
    if len(claims_dedup) != 0:
        combs = combinations(claims_dedup, 2)
        print("comb")
        from copy import deepcopy
        combs1 = deepcopy(combs)
        list_comb = list(combs)
        processed_comb = []
        while len(list_comb) != 0:
            comb = list_comb.pop(0)

            if comb in processed_comb:
                print('Already processed combination')
                continue
            else:
                processed_comb.append(comb)
            clm1 = comb[0]
            clm2 = comb[1]
            claim1 = " ".join(clm1)
            claim2 = " ".join(clm2)
            print("check similarity between---", claim1, "---", claim2)
            sem_result = check_semantic_similarity(claim1, claim2)
            print(sem_result)
            if (sem_result[0] == 'contradiction'):
                if len(claim1) > len(claim2):
                    print(claims_dedup)
                    claims_dedup.remove(clm2)
                    list_comb = list(combinations(claims_dedup, 2))
                    print("updated combs")
                else:
                    print(claims_dedup)
                    print(clm1)
                    claims_dedup.remove(clm1)
                    print(claims_dedup)
                    list_comb = list(combinations(claims_dedup, 2))
                    print("updated combs")

    #         if len(claims) == 0:
    #             print("Nothing identified by StandfordOpenIE")

    #             for sent in sents:
    #                 span = sent
    #                 #print(span)
    #                 chunks = list(span.noun_chunks)
    #                 #print(chunks)
    #                 comb = combinations(chunks, 2)
    #                 #print(list(comb))
    #                 for i in list(comb):
    #                     print(i[0])
    #                     print(i[1])
    #                     kg_item = ["","",""]
    #                     kg_item[0] = i0 = i[0].text
    #                     kg_item[2] = i1 = i[1].text
    #                     print(span.text)
    #                     result = re.search(rf"{i0}(.*?)?{i1}", span.text)
    #                     print("result",result)
    #                     if result != None:
    #                         kg_item[1] = result.group(1).strip()
    #                         print(kg_item)
    #                         claims_dedup.append(kg_item)
    print("Claims analyzed:")
    print(claims_dedup)
    return claims_dedup

def test_data_generator(sentence1, sentence2):
    sentence_pairs = np.array([[sentence1, sentence2]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    return test_data

def check_semantic_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[sentence1, sentence2]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    print(test_data)
    model1 = model
    proba = model1.predict(test_data)[0]
    model1 = ""
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}"
    pred = labels[idx]
    return pred, proba
#check_semantic_similarity("i am a boy", "I am a boy")
def find_entity(text):
    entity_dict = {}
    doc = nlp(text)
    for word in doc:
        # print(word, word.ent_type_ if word.ent_type_ != "" else "-")
        if word.ent_type_ != "":
            if word.ent_type_ in entity_dict:
                entity_dict[word.ent_type_] = [entity_dict[word.ent_type_], word.text]
            else:
                entity_dict[word.ent_type_] = word.text
    return entity_dict

def dict_compare(d1, d2):
    print(d1)
    print(d2)
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    # different_keys = added.union(removed,modified)
    return added, removed, modified, same

def compare_entity(claim, fact):
    import string
    from text2digits import text2digits
    t2d = text2digits.Text2Digits()

    claim = t2d.convert(claim.translate(claim.maketrans('', '', string.punctuation))) + "."
    fact = t2d.convert(fact.translate(fact.maketrans('', '', string.punctuation))) + "."
    claim_entity_dict = find_entity(claim)
    print("claim", claim)
    print(claim_entity_dict)
    fact_entity_dict = find_entity(fact)
    print("fact", fact)
    print(fact_entity_dict)
    added, removed, modified, same = dict_compare(claim_entity_dict, fact_entity_dict)
    for key in list(added):
        print('The claim has additional {key} information to the fact.'.format(key=key))
        print("The claim says", claim_entity_dict[key], ". The fact does not.")
    for key in list(removed):
        print('The claim does not have {key} information on the fact.'.format(key=key))
        print("The fact says", fact_entity_dict[key], ". The claim does not.")
    for key in list(modified):
        print('The claim has different {key} from the fact.'.format(key=key))
        print("Claim has", claim_entity_dict[key])
        print("Fact has", fact_entity_dict[key])
    for key in list(same):
        print('The claim has same {key} as the fact.'.format(key=key))
        print("Claim has", claim_entity_dict[key])
        print("Fact has", fact_entity_dict[key])
    accurate_req = ['GPE', 'ORDINAL', 'CARDINAL', "QUANTITY", "MONEY", "DATE", "TIME", "PERCENT"]
    reason = ""
    for key in list(modified):
        if key in accurate_req:
            # print(key, 'does not match, refuted.')
            reason = key + ' does not match, refuted. ' + "Claim has " + str(
                claim_entity_dict[key]) + ", but fact has " + str(fact_entity_dict[key])
            return True, reason
    return False, ""

import pandas as pd

df = pd.read_csv("./data/fact_base_test.csv")

def check_result_given_fact(input_stmt, fact):
    input_stmt = input_stmt.capitalize()
    fact = fact.capitalize()
    print("input stmt:", input_stmt)
    print("fact:", fact)
    overall_similarity = nlp(fact).similarity(nlp(input_stmt))
    print("overall similarity", overall_similarity )
    if overall_similarity < 0.7:
        print(("Fact does not relate to claim, cannot determine."))
        return (0, "Fact does not relate to claim, cannot determine.",input_stmt, fact)

    claims = process_claim(input_stmt)
    facts = process_claim(fact)
    df = pd.DataFrame(facts, columns=['subject', 'predicate', 'object'])
    print(df)
    spo_matched = []

    if len(claims) == 0 or len(facts) == 0:
        if len(claims) == 0:
            print("No atomic claim  extracted, comparing directly with given fact.")
        else:
            print("No atomic fact extracted, comparing directly with given claim.")
        result = check_semantic_similarity(fact, input_stmt)
        result2 = check_semantic_similarity(input_stmt, fact)
        print(result)
        print(result2)
        if (result[0] == 'contradiction') or (result2[0] == 'contradiction'):
            reason = "two statements are contradicting."
            print(0, str(reason), input_stmt, fact)
            return (0, reason, input_stmt, fact, input_stmt, fact)

        elif (result[0] != 'contradiction'):
            print("Result is not contradiction now check entity")
            refuted, reason = compare_entity(fact, input_stmt)
            if refuted == True:
                print(("Claim refuted by critical info.", str(reason), input_stmt, fact))
                return (0, str(reason), input_stmt, fact, input_stmt, fact)
            else:
                print("Atomic claim not refuted by critical info.")
                return (1, "The claim is valid", input_stmt, fact)
        return (0, "No claims or supported fact found for this claim, cannot determine.",input_stmt, fact )

    print("Now we check each atomic claim.")
    for claim in claims:
        potential_matched = []
        print("atomic claim under checking: ", claim)
        for idx, fact_row in df.iterrows():
            fact_atomic = fact_row['subject'] + " " + fact_row['predicate'] + " " + fact_row['object']
            print("atomic fact comparing:", fact_atomic)
            clm = " ".join(claim)
            atomic_similarity = nlp(fact_atomic).similarity(nlp(clm))
            print(atomic_similarity)
            if atomic_similarity >= 0.7:
                result = check_semantic_similarity(fact_atomic, clm)
                print(result)
                if (result[0] != 'contradiction') :
                    print("Result is not contradiction now check entity")
                    refuted, reason = compare_entity(clm, fact_atomic)
                    if refuted == True:
                        print(("Claim refuted by critical info.", str(reason), clm, fact_atomic))
                        return (0, str(reason), input_stmt, fact ,clm, fact_atomic)
                    else:
                        print("Atomic claim not refuted by critical info.")
                elif (result[0] == 'contradiction'):
                    reason = "two statements are contradicting."
                    print(0, str(reason), clm, fact_atomic)
                    return (0, reason,  input_stmt, fact ,clm, fact_atomic)

    return (1, "The claim is valid", input_stmt, fact)
    #     fact_match = pd.DataFrame(spo_matched, columns=['subject', 'predicate', 'object', 'confidence'])
    #
    # if len(fact_match) == 0:
    #     # print("No supported fact found for this claim, cannot determine.")
    #     print(0, "No supported fact found for this claim, cannot determine.", input_stmt)
    #     return (0, "No supported fact found for this claim, cannot determine.")
    # else:
    #     # no contradict with fact
    #     print(1, "The claim is valid", fact_match, input_stmt)
    #     return (1, "The claim is valid", fact_match)

def fact_check_result_naive(input_stmt, df):
    print(input_stmt)
    overall_similar = []
    for idx, fact_row in df.iterrows():
        fact = fact_row['original_sent']
        # print("fact comparing:", fact)
        fact = nlp(fact)
        overall_similarity = fact.similarity(nlp(input_stmt))
        # print("overall similarity", overall_similarity )
        if overall_similarity > 0.9:
            overall_similar.append(fact_row)
    print("overall_similar")
    print(overall_similar)
    if len(overall_similar) == 0:
        # print("No supported fact found for this claim, cannot determine.")
        return (0, "No supported fact found for this claim, cannot determine." ,input_stmt )

    else:
        for idx, fact_row in enumerate(overall_similar):
            fact = fact_row['subject'] + " " + fact_row['predicate'] + " " + fact_row['object']
            result = check_result_given_fact(input_stmt, fact)
            if result[0] == 0 :
                return result

    return (1, "The claim is valid", input_stmt, overall_similar)

test_set = pd.read_csv('./data/test_set.csv')

#fact_check_result_naive("Singapore is ranked 2nd on the Global Food Security Index.", df)
# pred_given = [(0, "DATE does not match, refuted. Claim has [[[['December', '2011'], 'January'], '17714400'], 'annual'], but fact has [[[['December', '2011'], 'January'], '177144000'], 'annual']", 'As of December 2011 and January 2013, 88000 foreigners and 5,400 Singaporeans were respectively diagnosed with HIV, but there are fewer than 10 annual deaths from HIV per 100,000 people.', 'As of December 2011 and January 2013, 8800 foreigners and 5,400 Singaporeans were respectively diagnosed with HIV, but there are fewer than 10 annual deaths from HIV per 100000 people.', 'As of December 2011 and January 2013, 88000 foreigners and 5,400 Singaporeans were respectively diagnosed with HIV, but there are fewer than 10 annual deaths from HIV per 100,000 people.', 'As of December 2011 and January 2013, 8800 foreigners and 5,400 Singaporeans were respectively diagnosed with HIV, but there are fewer than 10 annual deaths from HIV per 100000 people.'), (0, 'two statements are contradicting.', 'Education for primary, secondary, and tertiary levels is mostly supported by the private sectors.', 'Education for primary, secondary, and tertiary levels is mostly supported by the state.', 'Education is mostly supported by private sectors', 'Education is supported by state'), (0, 'two statements are contradicting.', 'Singapore is not home to ONE Championship, the biggest Mixed Martial Arts promotion in Asia.', 'Singapore is home to ONE Championship, the biggest Mixed Martial Arts promotion in Asia.', 'Singapore is not home to ONE Championship, the biggest Mixed Martial Arts promotion in Asia.', 'Singapore is home to ONE Championship, the biggest Mixed Martial Arts promotion in Asia.'), (0, 'two statements are contradicting.', 'Piracy in the Strait of Malacca has not been a cause of concern for all three countries.', 'Piracy in the Strait of Malacca has been a cause of concern for all three countries.', 'Piracy in the Strait of Malacca has not been a cause of concern for all three countries.', 'Piracy in the Strait of Malacca has been a cause of concern for all three countries.'), (0, 'two statements are contradicting.', 'In 1475 the Javanese king Kertanagara probably attacked Temasek when the Javanese king raided Pahang on the east coast of the Malay Peninsula.', 'In 1275 the Javanese king Kertanagara probably attacked Temasek when the Javanese king raided Pahang on the east coast of the Malay Peninsula.', 'king Kertanagara raided Pahang on east coast of Malay Peninsula', 'Javanese king Kertanagara raided Pahang on coast of Malay Peninsula'), (0, 'two statements are contradicting.', 'The armed forces of Singapore are divided into minions, air force, and navy branches.', 'The armed forces of Singapore are divided into army, air force, and navy branches.', 'armed forces are divided into minions air force', 'forces are divided'), (1, 'The claim is valid', 'The type of sand used in reclamation is found in rivers and deserts, rather than deserts, and is in great demand worldwide.', 'The type of sand used in reclamation is found in rivers and beaches, rather than deserts, and is in great demand worldwide.'), (0, 'two statements are contradicting.', '(Confrontation in Indonesian) in response to the formation of Singpore.', '(Confrontation in Indonesian) in response to the formation of Malaysia.', 'Confrontation is in response to formation of Singpore', 'Confrontation is in response to formation of Malaysia'), (1, 'The claim is valid', 'Singapore is ranked 2st on the Global Food Security Index.\n\n', 'Singapore is ranked 1st on the Global Food Security Index.\n\n'), (1, 'The claim is valid', "Singapore's well known gardens include the Singapore Botanic Gardens, a 162-year-old tropical garden and Singapore's first UNESCO World Heritage Site.\n\n", "Singapore's well known gardens include the Singapore Botanic Gardens, a 161-year-old tropical garden and Singapore's first UNESCO World Heritage Site.\n\n"), (0, 'two statements are contradicting.', '5,000 Malaysian students cross the Johor?�Singapore Causeway daily to attend schools in Malyasia.', '5,000 Malaysian students cross the Johor?�Singapore Causeway daily to attend schools in Singapore.', '5000 Malaysian students cross daily Johor Singapore Causeway', '5000 Malaysian students attend schools in Singapore'), (0, 'two statements are contradicting.', 'The plan was not for the Home Fleet to sail quickly to Singapore in the event of an emergency.', 'The plan was for the Home Fleet to sail quickly to Singapore in the event of an emergency.', 'The plan was not for the Home Fleet to sail quickly to Singapore in the event of an emergency.', 'The plan was for the Home Fleet to sail quickly to Singapore in the event of an emergency.'), (1, 'The claim is valid', "The RSAF's 130 Squadron is based in RAAF Base Pearce, Western Australia, and The RSAF's 130 Squadron 126 Squadron is based in the Conventional Centre, Queensland.", "The RSAF's 130 Squadron is based in RAAF Base Pearce, Western Australia, and The RSAF's 130 Squadron 126 Squadron is based in the Oakey Army Aviation Centre, Queensland."), (1, 'The claim is valid', 'During the 1990s when the National Arts Council was created to spearhead the development of performing arts, along with visual and literary art forms.', 'During the 1990s when the National Arts Council was created to spearhead the development of performing arts, along with visual and literary art forms.'), (0, 'two statements are contradicting.', 'Singapore, one of the great trading entrepots of the British empire, has experienced bad economic growth and diversification since 1960.', 'Singapore, one of the great trading entrepôts of the British empire, has experienced remarkable economic growth and diversification since 1960.', 'Singapore has experienced bad economic growth', 'Singapore has experienced diversification since 1960'), (0, 'two statements are contradicting.', 'The ruling party of Malaya, United Malays National Organisation (UMNO), was staunchly anti-communist, and The ruling party of Malaya, United Malays National Organisation (UMNO) was suspected UMNO would support the communist factions of PAP.', 'The ruling party of Malaya, United Malays National Organisation (UMNO), was staunchly anti-communist, and The ruling party of Malaya, United Malays National Organisation (UMNO) was suspected UMNO would support the non-communist factions of PAP.', 'ruling party ruling party of United Malays National Organisation', 'party was Malaya United Malays National Organisation UMNO Malaya United Malays National Organisation UMNO'), (1, 'The claim is valid', 'Winston Churchill touted the large Singapore Naval Base as the "Gibraltar of the East", and military discussions often referred to the large Singapore Naval Base as simply "East of Singapore".', 'Winston Churchill touted the large Singapore Naval Base as the "Gibraltar of the East", and military discussions often referred to the large Singapore Naval Base as simply "East of Suez".'), (0, "DATE does not match, refuted. Claim has ['recent', 'years'], but fact has years", 'Buddhism has also made fast inroads into the country in recent years.\n\n', 'Buddhism has also made slow inroads into the country in recent years.\n\n', 'Buddhism has also made inroads in recent years', 'Buddhism has also made inroads in years'), (0, 'two statements are contradicting.', 'In the 2016 EF English Proficiency Index taken in 72 countries, Singapore place 10th and has been the only Asian country in the top ten.\n\n', 'In the 2016 EF English Proficiency Index taken in 72 countries, Singapore place 6th and has been the only Asian country in the top ten.\n\n', '2016 EF English Proficiency Index taken in 72 countries', '2016 EF English Proficiency Index taken Singapore place 6th'), (1, 'The claim is valid', 'In 1888, the colonies in East Asia were reorganised and Singapore came under the direct control of Britain as part of the Straits Settlements.', 'In 1867, the colonies in East Asia were reorganised and Singapore came under the direct control of Britain as part of the Straits Settlements.'), (1, 'The claim is valid', "Starhub Cable Vision (SCV) also offers cable television with channels from all around the world, and Singtel's Mio TV provides an cable service.", "Starhub Cable Vision (SCV) also offers cable television with channels from all around the world, and Singtel's Mio TV provides an IPTV service."), (0, 'two statements are contradicting.', "English is largely based on American English, owing to the country's status as a former crown colony.", "English is largely based on British English, owing to the country's status as a former crown colony.", 'English is based on American English', 'English is based on British English'), (1, 'The claim is valid', 'A series of strikes in 1950 caused massive stoppages in public transport and other services.', 'A series of strikes in 1947 caused massive stoppages in public transport and other services.'), (1, 'The claim is valid', 'The Chinese traveller Wang Dayuan visited a place around 1990 named Danmaxi  Tan Ma Hsi) or Tam ma siak, depending on pronunciation.', 'The Chinese traveller Wang Dayuan visited a place around 1330 named Danmaxi (Chinese: 淡馬?? pinyin: D?nm?xí; Wade?�Giles: Tan Ma Hsi) or Tam ma siak, depending on pronunciation.'), (1, 'The claim is valid', 'Cultural activities in Singapore are not derivative, springing from one or another of the major civilizations of China, India, Indonesia, or the West.', 'Cultural activities in Singapore are largely derivative, springing from one or another of the major civilizations of China, India, Indonesia, or the West.'), (0, "CARDINAL does not match, refuted. Claim has ['over', '10'], but fact has ['over', '100']", 'Changi Airport hosts a network of over 10 airlines connecting Singapore to some 300 cities in about 70 countries and territories worldwide.', 'Changi Airport hosts a network of over 100 airlines connecting Singapore to some 300 cities in about 70 countries and territories worldwide.', 'Changi Airport hosts network of over 10 airlines', 'Changi Airport hosts network of over 100 airlines'), (0, 'two statements are contradicting.', 'Literature of Singapore, or "SingLit", comprises a collection of literary works by Singaporeans written chiefly in the country\'s four official languages: English, Malay, Cantonese, and Tamil.', 'Literature of Singapore, or "SingLit", comprises a collection of literary works by Singaporeans written chiefly in the country\'s four official languages: English, Malay, Mandarin, and Tamil.', 'Literature comprises collection by Singaporeans written in countrys four official languages English Malay Cantonese', 'Literature comprises collection by Singaporeans written chiefly in countrys four official languages English Malay Mandarin'), (0, 'two statements are contradicting.', 'To overcome this problem, the government has not been encouraging foreigners to immigrate to Singapore for the past few decades.', 'To overcome this problem, the government has been encouraging foreigners to immigrate to Singapore for the past few decades.', 'To overcome this problem, the government has not been encouraging foreigners to immigrate to Singapore for the past few decades.', 'To overcome this problem, the government has been encouraging foreigners to immigrate to Singapore for the past few decades.'), (0, 'two statements are contradicting.', 'The government Monetary Authority of Singapore performs none of the functions of a central bank except issuing currency.', 'The government?�s Monetary Authority of Singapore performs all the functions of a central bank except issuing currency.', 'government Monetary Authority performs none of functions of central bank', 's Monetary Authority issuing currency'), (0, 'CARDINAL does not match, refuted. Claim has 2, but fact has 4', 'Singapore is increasingly regarded as having two sub-literatures instead of one.', 'Singapore is increasingly regarded as having four sub-literatures instead of one.', 'Singapore having two sub-literatures', 'Singapore having four sub-literatures'), (0, "CARDINAL does not match, refuted. Claim has ['about', '1'], but fact has ['about', '3']", 'The proportion of Christians, Taoists, and non-religious people increased between 2000 and 2010 by about 1 percentage points each, while the proportion of Buddhists decreased.', 'The proportion of Christians, Taoists, and non-religious people increased between 2000 and 2010 by about 3 percentage points each, while the proportion of Buddhists decreased.', 'proportion increased by about 1 percentage points', 'proportion increased by about 3 percentage points'), (0, "CARDINAL does not match, refuted. Claim has ['some', '1000'], but fact has 8000", "The National Gallery Singapore is the nation's flagship museum with some 1,000 works from Singaporean and other Southeast Asian artists.", "The National Gallery Singapore is the nation's flagship museum with some 8,000 works from Singaporean and other Southeast Asian artists.", "National Gallery Singapore is nation 's flagship museum with some 1,000 works from Singaporean", 'nation has flagship museum with some 8,000 works from Singaporean'), (0, 'two statements are contradicting.', 'He resigned and was replaced by Lim Yew Hock in 1956, and after further negotiations Britain to grant Singapore full internal self-government for all matters except defence and monetary.', 'He resigned and was replaced by Lim Yew Hock in 1956, and after further negotiations Britain to grant Singapore full internal self-government for all matters except defence and foreign affairs.', 'He was replaced in 1956', 'He was replaced'), (0, 'two statements are contradicting.', 'No members of a racial community may qualify as candidates in a reserved presidential election.\n\n', 'Only members of a racial community may qualify as candidates in a reserved presidential election.\n\n', 'No members of a racial community may qualify as candidates in a reserved presidential election.\n\n', 'Only members of a racial community may qualify as candidates in a reserved presidential election.\n\n'), (1, 'The claim is valid', 'The Israeli Defense Force (IDF) commanders were tasked by the government to create the Singapore Armed Forces (SAF) from scratch, and Israeli instructors were brought in to train Malaysian soldiers.', 'The Israeli Defense Force (IDF) commanders were tasked by the government to create the Singapore Armed Forces (SAF) from scratch, and Israeli instructors were brought in to train Singaporean soldiers.'), (0, 'two statements are contradicting.', 'Pre-university education takes place over five to six years at senior schools, mostly called Junior Colleges.', 'Pre-university education takes place over two to three years at senior schools, mostly called Junior Colleges.', 'Pre-university education takes place at senior schools', 'Preuniversity education takes place at senior schools called'), (1, 'The claim is valid', 'Singapore, city-state located at the northern tip of the Malay Peninsula, about 85 miles (137 kilometres)', 'Singapore, city-state located at the southern tip of the Malay Peninsula, about 85 miles (137 kilometres)'), (0, 'two statements are contradicting.', "While elections are considered generally free, the Singaporean government exercises no control over politics and society, and the People's Action Party has ruled continuously since independence.", "While elections are considered generally free, the Singaporean government exercises significant control over politics and society, and the People's Action Party has ruled continuously since independence.", "People 's Action Party has ruled continuously since independence", 'government exercises control over politics'), (1, 'The claim is valid', "Border issues exist with Malaysia and China, and both have banned the sale of marine sand to Singapore over disputes about Singapore's land reclamation.", "Border issues exist with Malaysia and Indonesia, and both have banned the sale of marine sand to Singapore over disputes about Singapore's land reclamation."), (0, 'CARDINAL does not match, refuted. Claim has 3, but fact has 2', 'In retaliation, Singapore did not extend to Sabah and Sarawak the full extent of the loans agreed to for economic development of the three eastern states.', 'In retaliation, Singapore did not extend to Sabah and Sarawak the full extent of the loans agreed to for economic development of the two eastern states.', 'full extent agreed to to development of three states', 'extent agreed to to economic development of two eastern states'), (1, 'The claim is valid', 'For several years, Singapore has not been one of the few countries with an AAA credit rating from the big three, and the only Asian country to achieve this rating.', 'For several years, Singapore has been one of the few countries with an AAA credit rating from the big three, and the only Asian country to achieve this rating.'), (1, 'The claim is valid', 'Singapore students have excelled in many of the world education benchmarks in maths, science and creativity.', 'Singapore students have excelled in many of the world education benchmarks in maths, science and reading.'), (0, "GPE does not match, refuted. Claim has China, but fact has ['China', 'China']", 'Mandarin, the official language of China, transcends dialect barriers, and China use is not promoted; one-third of the school population is taught in that language.', 'Mandarin, the official language of China, transcends dialect barriers, and China use is strongly promoted; one-third of the school population is taught in that language.', 'Mandarin official language of China', 'China use Mandarin official language of China'), (1, 'The claim is valid', ': Parliament does not enacts national law, approves budgets, and provides a check on government policy.\n', ': Parliament enacts national law, approves budgets, and provides a check on government policy.\n'), (1, 'The claim is valid', 'In 2010, Singapore was removed from the Organisation for Economic Co-operation and Development (OECD) "liste grise" of tax havens, but ranked fourth on the Tax Justice Network\'s 2015 Financial Secrecy Index of the world\'s off-shore financial service providers, banking one-eighth of the world\'s offshore capital, while "providing numerous tax avoidance and evasion opportunities".', 'In 2009, Singapore was removed from the Organisation for Economic Co-operation and Development (OECD) "liste grise" of tax havens, but ranked fourth on the Tax Justice Network\'s 2015 Financial Secrecy Index of the world\'s off-shore financial service providers, banking one-eighth of the world\'s offshore capital, while "providing numerous tax avoidance and evasion opportunities".'), (0, 'two statements are contradicting.', 'Close economic ties exist with Brunei, and the two does not share a pegged currency value, through a Currency Interchangeability Agreement between the two countries which makes both Brunei dollar and Singapore dollar banknotes and coins legal tender in either country.\n\n', 'Close economic ties exist with Brunei, and the two share a pegged currency value, through a Currency Interchangeability Agreement between the two countries which makes both Brunei dollar and Singapore dollar banknotes and coins legal tender in either country.\n\n', 'Close economic ties exist with Brunei, and the two does not share a pegged currency value, through a Currency Interchangeability Agreement between the two countries which makes both Brunei dollar and Singapore dollar banknotes and coins legal tender in either country.\n\n', 'Close economic ties exist with Brunei, and the two share a pegged currency value, through a Currency Interchangeability Agreement between the two countries which makes both Brunei dollar and Singapore dollar banknotes and coins legal tender in either country.\n\n'), (1, 'The claim is valid', 'Singapore has many natural resources.', 'Singapore has few natural resources.'), (0, 'CARDINAL does not match, refuted. Claim has 8, but fact has 20', 'Singapore has eight public universities of which the National University of Singapore and Nanyang Technological University are among the top 20 universities in the world.\n\n', 'Singapore has six public universities of which the National University of Singapore and Nanyang Technological University are among the top 20 universities in the world.\n\n', 'Singapore has eight public universities', 'top 20 universities is in world'), (1, 'The claim is valid', "Racial and religious harmony is regarded by Singaporeans as a crucial part of Singapore's success, and played no part in building a Singaporean identity.\n\n", "Racial and religious harmony is regarded by Singaporeans as a crucial part of Singapore's success, and played a part in building a Singaporean identity.\n\n"), (0, 'two statements are contradicting.', 'These are staffed by civil servants who are not monitored by an independent Public Service Commission.', 'These are staffed by civil servants who are monitored by an independent Public Service Commission.', 'These are staffed by civil servants who are not monitored by an independent Public Service Commission.', 'These are staffed by civil servants who are monitored by an independent Public Service Commission.'), (1, 'The claim is valid', 'It is used in the national anthem Little Stars��? in citations of Singaporean orders and decorations and in military commands.', 'It is used in the national anthem ?�’Majulah Singapura?��? in citations of Singaporean orders and decorations and in military commands.'), (0, 'CARDINAL does not match, refuted. Claim has 3, but fact has 2', "The country's territory has the third greatest population density in the world.", "The country's territory has the second greatest population density in the world.", "country 's territory has third population density", "country 's territory has second greatest population density in world"), (0, "QUANTITY does not match, refuted. Claim has [['about', '1'], 'degree'], but fact has [[[[[['about', '1'], 'degree'], '137'], 'kilometres'], '85'], 'miles']", 'a sovereign island city-state in maritime Southeast Asia lies about one degree of latitude (147 kilometres or 85 miles) north of the equator, off the southern tip of the Malay Peninsula, bordering the Straits of Malacca (Melaka) to the West, the Riau Islands (Indonesia) to the South, and the South China Sea to the east.', 'a sovereign island city-state in maritime Southeast Asia lies about one degree of latitude (137 kilometres or 85 miles) north of the equator, off the southern tip of the Malay Peninsula, bordering the Straits of Malacca (Melaka) to the West, the Riau Islands (Indonesia) to the South, and the South China Sea to the east.', 'sovereign island city-state lies about one degree', 'island citystate lies about one degree of latitude 137 kilometres 85 miles north'), (0, 'two statements are contradicting.', 'After hearing rumours that they were to be sent to fight the Ottoman Empire, a Muslim state, the soldiers rebelled, killing the soldiers officers and several British civilians after the mutiny was suppressed by non-Muslim troops arriving from Johore and Burma.\n\n', 'After hearing rumours that they were to be sent to fight the Ottoman Empire, a Muslim state, the soldiers rebelled, killing the soldiers officers and several British civilians before the mutiny was suppressed by non-Muslim troops arriving from Johore and Burma.\n\n', 'soldiers killing soldiers officers suppressed by non-Muslim troops arriving from Johore', 'mutiny was suppressed by nonMuslim troops arriving from Johore'), (0, 'CARDINAL does not match, refuted. Claim has 70, but fact has 60', 'In 2019, there were more than 70 semiconductor companies in Singapore, which together constituted 11% of the global market share.', 'In 2019, there were more than 60 semiconductor companies in Singapore, which together constituted 11% of the global market share.', '70 semiconductor companies is in Singapore', '60 semiconductor companies is in Singapore'), (1, 'The claim is valid', "the People's Action Party (PAP) occupies a dominant position in Singaporean politics, having won large parliamentary majorities in every election since self-governance was granted in 1949.", "the People's Action Party (PAP) occupies a dominant position in Singaporean politics, having won large parliamentary majorities in every election since self-governance was granted in 1959."), (1, 'The claim is valid', 'Internet in Singapore is provided by state owned Singtel, partially state owned Starhub and M1 Limited as well as some other business internet service providers (ISPs) that offer residential service plans of speeds up to 2 Gbit/s as of spring 2016.', 'Internet in Singapore is provided by state owned Singtel, partially state owned Starhub and M1 Limited as well as some other business internet service providers (ISPs) that offer residential service plans of speeds up to 2 Gbit/s as of spring 2015.'), (0, 'two statements are contradicting.', "Singaporeans are mostly bilingual, typically with English as Singaporeans common language and Singaporeans mother-tongue as a second language taught in schools, in order to preserve each individual's ethnic identity and money.", "Singaporeans are mostly bilingual, typically with English as Singaporeans common language and Singaporeans mother-tongue as a second language taught in schools, in order to preserve each individual's ethnic identity and values.", 'Singaporeans are mostly bilingual with typically English as Singaporeans common language as second language taught in schools', 'Singaporeans are mostly bilingual with English as Singaporeans common language as second language taught in schools'), (0, 'two statements are contradicting.', 'About two-thirds of all Chinese profess some degree of attachment to Confucianism, Christianity, or Daoism or to some combination thereof.', 'About two-thirds of all Chinese profess some degree of attachment to Confucianism, Buddhism, or Daoism or to some combination thereof.', 'About two-thirds of all Chinese profess some degree of attachment to Confucianism, Christianity, or Daoism or to some combination thereof.', 'About two-thirds of all Chinese profess some degree of attachment to Confucianism, Buddhism, or Daoism or to some combination thereof.'), (0, "DATE does not match, refuted. Claim has ['years', '2012'], but fact has ['years', '2010']", 'While Singapore is not a formal member of the G20, it has been invited to participate in the G20 processes in most years since 2012.', 'While Singapore is not a formal member of the G20, it has been invited to participate in the G20 processes in most years since 2010.', 'it participate in years since 2012', 'it participate in years since 2010'), (0, 'two statements are contradicting.', 'The eastern part of the main island is a high plateau cut by erosion into an intricate pattern of hills and valleys.', 'The eastern part of the main island is a low plateau cut by erosion into an intricate pattern of hills and valleys.', 'main island eastern part of is high plateau cut into intricate pattern of hills', 'main island eastern part of is low plateau cut by erosion into intricate pattern'), (0, 'two statements are contradicting.', 'There are two-man-made connections to Johor, Malaysia: the Woodlands 1st Link in the south and the Tuas 2nd Link in the west.', 'There are two-man-made connections to Johor, Malaysia: the Woodlands 1st Link in the north and the Tuas 2nd Link in the west.', 'Tuas 2nd Link is in west', 'Woodlands 1st Link is in north'), (0, 'CARDINAL does not match, refuted. Claim has 2, but fact has 1', 'It was the inaugural F1 night race, and the second F1 street race in Asia.', 'It was the inaugural F1 night race, and the first F1 street race in Asia.', 'second F1 street race is in Asia', 'first F1 street race is in Asia'), (1, 'The claim is valid', 'north of the Equator consists of the diamond-shaped Singapore Island and some 70 small islets; the main island occupies all but about 18 square miles of this combined area.', 'north of the Equator consists of the diamond-shaped Singapore Island and some 60 small islets; the main island occupies all but about 18 square miles of this combined area.'), (1, 'The claim is valid', 'From 2012 onward, people may register using a multi-racial classification, in which people may choose one primary race and one secondary race, but no more than two.\n\n', 'From 2010 onward, people may register using a multi-racial classification, in which people may choose one primary race and one secondary race, but no more than two.\n\n'), (0, 'two statements are contradicting.', 'It is interchangeable with the Singapore dollar (SGD or S$), issued by the Monetary Authority of Singapore (MAS) at par value since 1997.', 'It is interchangeable with the Singapore dollar (SGD or S$), issued by the Monetary Authority of Singapore (MAS) at par value since 1967.', 'It is interchangeable with Singapore dollar issued by Monetary Authority of Singapore at value since 1997', 'It is interchangeable with Singapore dollar SGD issued by Monetary Authority at par value since 1967'), (1, 'The claim is valid', 'A surplus of food led to malnutrition, disease, and rampant crime and violence.', 'A shortage of food led to malnutrition, disease, and rampant crime and violence.'), (0, 'GPE does not match, refuted. Claim has Singapura, but fact has Palembang', 'The semi-historical Malay Annals state that Temasek was christened Singapura by Sang Nila Utama, a 14th-century Srivijayan Raja from Palembang.', 'The semi-historical Malay Annals state that Temasek was christened Singapura by Sang Nila Utama, a 13th-century Srivijayan Raja from Palembang.', 'Temasek was christened Singapura by Sang Nila Utama', 'Sang Nila Utama Raja from Palembang'), (1, 'The claim is valid', 'Singapore is the 1st most visited city in the world, and 2nd in the Asia-Pacific.', 'Singapore is the 5th most visited city in the world, and 2nd in the Asia-Pacific.'), (1, 'The claim is valid', 'China relationship,[failed verification] and has maintained a long-standing and greatly prioritised close relationship partly due to China\'s growing influence and essentiality in the Asia-Pacific region, specifying that "China relationship,[failed common interest with China is far smaller than any differences".', 'China relationship,[failed verification] and has maintained a long-standing and greatly prioritised close relationship partly due to China\'s growing influence and essentiality in the Asia-Pacific region, specifying that "China relationship,[failed common interest with China is far greater than any differences".'), (0, 'two statements are contradicting.', 'Singapore, city-state located at the southern tip of the Malay Peninsula, about 185 miles (137 kilometres) north of the Equator.', 'Singapore, city-state located at the southern tip of the Malay Peninsula, about 85 miles (137 kilometres) north of the Equator.', 'Singapore city-state north about 185 miles', 'Singapore city-state north about 85 miles'), (1, 'The claim is valid', 'Singapore sailors have had success on the international stage, with Singapore sailors Optimist team being considered among the worst in the world.', 'Singapore sailors have had success on the international stage, with Singapore sailors Optimist team being considered among the best in the world.'), (0, 'two statements are contradicting.', 'Singapore is not a member of the United Nations, World Trade Organization, East Asia Summit, Non-Aligned Movement and the Commonwealth of Nations.\n\n', 'Singapore is also a member of the United Nations, World Trade Organization, East Asia Summit, Non-Aligned Movement and the Commonwealth of Nations.\n\n', 'Singapore is not a member of the United Nations, World Trade Organization, East Asia Summit, Non-Aligned Movement and the Commonwealth of Nations.\n\n', 'Singapore is also a member of the United Nations, World Trade Organization, East Asia Summit, Non-Aligned Movement and the Commonwealth of Nations.\n\n'), (1, 'The claim is valid', 'Another name, the "Little Red Dot", was adopted after an article publication in the Asian Wall Street Journal of 4 August 1998 regarded the third Indonesian President B. J. Habibie by referring to Indonesia as a red dot on a map.\n\n', 'Another name, the "Little Red Dot", was adopted after an article publication in the Asian Wall Street Journal of 4 August 1998 regarded the third Indonesian President B. J. Habibie by referring to Singapore as a red dot on a map.\n\n'), (0, 'two statements are contradicting.', 'Mandarin is the lingua franca and the main language used in business, government, law and education.', 'English is the lingua franca and the main language used in business, government, law and education.', 'Mandarin language used in business', 'English language used in business'), (0, "GPE does not match, refuted. Claim has ['Indonesia', 'Singapore'], but fact has ['Indonesia', 'Malaysia']", 'Indonesia opposed the formation of Singapore due to Indonesia own claims over North Borneo and launched konfrontasi', 'Indonesia opposed the formation of Malaysia due to Indonesia own claims over North Borneo and launched konfrontasi', 'Indonesia opposed formation of Singapore', 'Indonesia opposed formation of Malaysia'), (0, 'CARDINAL does not match, refuted. Claim has 1, but fact has 3', "Cabinet is chosen by the country's first prime minister and formally appointed by Lee Kuan Yew.\n\n", "Cabinet is chosen by the country's third prime minister and formally appointed by Lee Kuan Yew.\n\n", 'country by first prime minister', 'Cabinet is chosen by countrys third minister'), (0, "GPE does not match, refuted. Claim has ['Singapore', 'Malaysia'], but fact has ['Singapore', 'Singapore']", 'Singapore promotes Malaysia as a medical tourism hub, with about 200,000 foreigners seeking medical care there each year.', 'Singapore promotes Singapore as a medical tourism hub, with about 200,000 foreigners seeking medical care there each year.', 'Singapore promotes Malaysia as tourism hub', 'Singapore promotes Singapore'), (0, 'two statements are contradicting.', 'Singapore began hosting a round of the Formula One World Championship, the Singapore Grand Prix at the Marina Bay Street Circuit in 2001.', 'Singapore began hosting a round of the Formula One World Championship, the Singapore Grand Prix at the Marina Bay Street Circuit in 2008.', 'hosting round of Formula One World Championship', 'Singapore hosting round in 2008'), (0, 'two statements are contradicting.', 'The national airline is not Singapore Airlines.', 'The national airline is Singapore Airlines.', 'The national airline is not Singapore Airlines.', 'The national airline is Singapore Airlines.'), (1, 'The claim is valid', 'Singapore a "garden city" is not a popular location for conferences and events.\n\n', 'Singapore a "garden city" is a popular location for conferences and events.\n\n'), (0, 'two statements are contradicting.', 'British Military Administration ended on 1 May 1946, with Singapore becoming a separate Crown Colony.', 'British Military Administration ended on 1 April 1946, with Singapore becoming a separate Crown Colony.', 'British Military Administration ended Singapore becoming', 'Singapore becoming Crown Colony'), (0, 'DATE does not match, refuted. Claim has annually, but fact has July', "The Singapore Food Festival which celebrates Singapore's cuisine is held annually in June.\n\n", "The Singapore Food Festival which celebrates Singapore's cuisine is held annually in July.\n\n", 'Singapore Food Festival is held annually', 'Singapore Food Festival is held in July'), (1, 'The claim is valid', 'Because of the scarcity of open land on the main island, training involving activities such as live firing and amphibious warfare are often carried out on larger islands, typically barred to civilian access.', 'Because of the scarcity of open land on the main island, training involving activities such as live firing and amphibious warfare are often carried out on smaller islands, typically barred to civilian access.'), (1, 'The claim is valid', 'As alternatives to Pre-U education, however, courses are offered in other post-secondary education institutions, including 5 polytechnics and the Institutes of Technical Education (ITEs).', 'As alternatives to Pre-U education, however, courses are offered in other post-secondary education institutions, including 5 polytechnics and the Institutes of Technical Education (ITEs).'), (1, 'The claim is valid', 'An underlying principle is political and economic stability in the region.', 'An underlying principle is political and economic stability in the region.'), (1, 'The claim is valid', 'a means "lion", pura means "country" or "fortress").', '; siṃha means "lion", pura means "city" or "fortress").'), (1, 'The claim is valid', 'Economic development has not been closely supervised by the Singaporean government, and Economic development has been highly dependent on investment capital from foreign multinational corporations.', 'Economic development has been closely supervised by the Singaporean government, and Economic development has been highly dependent on investment capital from foreign multinational corporations.'), (0, 'DATE does not match, refuted. Claim has 1890, but fact has 1867', 'In 1890 they were made a crown colony under the Colonial Office in London.\n\n', 'In 1867 they were made a crown colony under the Colonial Office in London.\n\n', 'they were made crown colony In 1890', 'they were made crown colony In 1867'), (0, "GPE does not match, refuted. Claim has ['Singapore', 'Singapore'], but fact has Singapore", 'Singapore\'s unique combination of a strong almost authoritarian government with an emphasis on meritocracy and good governance is known as the "Chinese model", and is regarded as a key factor behind Singapore\'s political stability, economic growth, and harmonious social order.', 'Singapore\'s unique combination of a strong almost authoritarian government with an emphasis on meritocracy and good governance is known as the "Singapore model", and is regarded as a key factor behind Singapore\'s political stability, economic growth, and harmonious social order.', "Singapore 's unique combination is regarded as factor behind Singapore 's political stability", 'Singapore has unique combination of strong almost authoritarian government with emphasis on meritocracy'), (1, 'The claim is valid', "32% of healthcare accounts for approximately 13.5% of Singapore's GDP.\n\n", "32% of healthcare accounts for approximately 3.5% of Singapore's GDP.\n\n"), (0, 'two statements are contradicting.', 'The older parts of The city of Singapore have not been substantially refurbished, especially along the Singapore River but elsewhere as well.', 'The older parts of The city of Singapore have been substantially refurbished, especially along the Singapore River but elsewhere as well.', 'The older parts of The city of Singapore have not been substantially refurbished, especially along the Singapore River but elsewhere as well.', 'The older parts of The city of Singapore have been substantially refurbished, especially along the Singapore River but elsewhere as well.'), (0, "QUANTITY does not match, refuted. Claim has ['7356', 'kilometres'], but fact has [['kilometres', '100'], 'mi']", 'Upgraded in 1998 and renamed Electronic Road Pricing, a road system covering 7,356 kilometres (2,085 mi), which includes 161 kilometres (100 mi) of expressways introduced electronic toll collection, electronic detection, and video surveillance technology.', 'Upgraded in 1998 and renamed Electronic Road Pricing, a road system covering 3,356 kilometres (2,085 mi), which includes 161 kilometres (100 mi) of expressways introduced electronic toll collection, electronic detection, and video surveillance technology.', 'road system covering 7,356 kilometres', 'expressways of kilometres is 100 mi'), (1, 'The claim is valid', 'Despite its size, the country has not dominated swim meets in the Southeast Asia Games.', 'Despite its size, the country has dominated swim meets in the Southeast Asia Games.'), (0, "GPE does not match, refuted. Claim has ['Singapore', 'Malaysia'], but fact has Singapore", 'The currency of Singapore is the Singapore dollar (SGD or S$), issued by the Monetary Authority of Malaysia (MAS).', 'The currency of Singapore is the Singapore dollar (SGD or S$), issued by the Monetary Authority of Singapore (MAS).', 'Singapore dollar issued by Monetary Authority Malaysia', 'Singapore dollar issued by Monetary Authority'), (0, 'two statements are contradicting.', "The Port of Singapore became one of the world's busiest ports and the service and tourism industries also did not grow immensely during this period.\n\n", "The Port of Singapore became one of the world's busiest ports and the service and tourism industries also grew immensely during this period.\n\n", "The Port of Singapore became one of the world's busiest ports and the service and tourism industries also did not grow immensely during this period.\n\n", "The Port of Singapore became one of the world's busiest ports and the service and tourism industries also grew immensely during this period.\n\n"), (0, 'two statements are contradicting.', 'The Constitution is not the supreme law of the country, establishing the structure and responsibility of government.', 'The Constitution is the supreme law of the country, establishing the structure and responsibility of government.', 'The Constitution is not the supreme law of the country, establishing the structure and responsibility of government.', 'The Constitution is the supreme law of the country, establishing the structure and responsibility of government.'), (0, 'two statements are contradicting.', 'Various communities have Various communities own distinct ethnic musical traditions: Japanese, Malays, Indians, and Eurasians.', 'Various communities have Various communities own distinct ethnic musical traditions: Chinese, Malays, Indians, and Eurasians.', 'Various communities own distinct traditions Japanese Malays Indians', 'Various communities own distinct traditions'), (0, 'two statements are contradicting.', 'MediaCorp operates no free-to-air television channels and free-to-air radio stations in Singapore.', 'MediaCorp operates most free-to-air television channels and free-to-air radio stations in Singapore.', 'MediaCorp operates no free-to-air television channels and free-to-air radio stations in Singapore.', 'MediaCorp operates most free-to-air television channels and free-to-air radio stations in Singapore.'), (0, 'CARDINAL does not match, refuted. Claim has 4, but fact has 5', "Five companies, Go-Ahead, Tower-Transit, SBS Transit and SMRT Corporation run the public buses under a 'Bus Contracting Model' where operators bid for routes.", "Four companies, Go-Ahead, Tower-Transit, SBS Transit and SMRT Corporation run the public buses under a 'Bus Contracting Model' where operators bid for routes.", "Five companies, Go-Ahead, Tower-Transit, SBS Transit and SMRT Corporation run the public buses under a 'Bus Contracting Model' where operators bid for routes.", "Four companies, Go-Ahead, Tower-Transit, SBS Transit and SMRT Corporation run the public buses under a 'Bus Contracting Model' where operators bid for routes."), (1, 'The claim is valid', 'A sizeable minority of middle-class known as Peranakans or Baba-Nyonya descendants of 15th- and 16th-century Chinese immigrants was here.', 'There was also a sizeable minority of middle-class, locally born people known as Peranakans or Baba-Nyonya descendants of 15th- and 16th-century Chinese immigrants.'), (1, 'The claim is valid', 'The Singaporean economy top contributors are financial services, manufacturing, and oil-refining.', 'The Singaporean economy is diversified, with The Singaporean economy top contributors being financial services, manufacturing, and oil-refining.'), (1, 'The claim is valid', 'The occupation was to become a major turning point in the histories of several nations, including those of Japan, Britain, and Singapore.', 'The occupation was to become a major turning point in the histories of several nations, including those of Japan, Britain, and Singapore.'), (1, 'The claim is valid', 'Talks soon broke down, and abusive speeches and writing became rife on both sides.', 'Talks soon broke down, and abusive speeches and writing became rife on both sides.'), (0, 'two statements are contradicting.', 'Singapore has 4 official languages.', 'Singapore has four official languages: English, Malay, Mandarin, and Tamil.', 'Singapore has 4 languages', 'Singapore has four languages'), (1, 'The claim is valid', "As of 2018 Singapore's ranking in the Human Development Index is ninth in the world, with an HDI value of 0.935.\n\n", "As of 2018 Singapore's ranking in the Human Development Index is 9th in the world, with an HDI value of 0.935.\n\n"), (1, 'The claim is valid', "Land reclamation projects had increased Singapore's land area from 580 km2 (220 sq mi) in the 1960s to 710 km2 ", "Land reclamation projects had increased Singapore's land area from 580 km2 (220 sq mi) in the 1960s to 710 km2 (270 sq mi) by 2015, an increase of some 22% (130 km2)."), (0, "DATE does not match, refuted. Claim has [[['1830', '2'], 'years'], 'later'], but fact has 1830", 'In 1830 Penang, and Malacca (Melaka) were reduced to a residency under Bengal.', 'In 1830 Penang, and Malacca (Melaka) were reduced to a residency under Bengal, and two years later Singapore became Penang, and Malacca (Melaka) capital.', 'In 1830 Penang, and Malacca (Melaka) were reduced to a residency under Bengal.', 'In 1830 Penang, and Malacca (Melaka) were reduced to a residency under Bengal, and two years later Singapore became Penang, and Malacca (Melaka) capital.'), (1, 'The claim is valid', 'British, Australian, and Indian troops led by Lord Louis Mountbatten returned to Singapore to receive the formal surrender of Japanese forces in the region from General Itagaki Seishiro on behalf of General Hisaichi Terauchi on 12 September 1945.', 'British, Australian, and Indian troops led by Lord Louis Mountbatten returned to Singapore to receive the formal surrender of Japanese forces in the region from General Itagaki Seishiro on behalf of General Hisaichi Terauchi on 12 September 1945.'), (0, "GPE does not match, refuted. Claim has [[[[[['Hong', 'Kong'], 'South'], 'Korea'], 'Taiwan'], 'Singapore'], 'Singapore'], but fact has [[[[['Hong', 'Kong'], 'South'], 'Korea'], 'Taiwan'], 'Singapore']", 'Along with Hong Kong, South Korea, and Taiwan, Singapore is one of the Four Asian Tigers.', 'Along with Hong Kong, South Korea, and Taiwan, Singapore is one of the Four Asian Tigers, but has surpassed Singapore peers in terms of Gross Domestic Product (GDP) per capita.', 'Along with Hong Kong, South Korea, and Taiwan, Singapore is one of the Four Asian Tigers.', 'Along with Hong Kong, South Korea, and Taiwan, Singapore is one of the Four Asian Tigers, but has surpassed Singapore peers in terms of Gross Domestic Product (GDP) per capita.'), (0, 'two statements are contradicting.', 'Private ownership of TV satellite dishes is banned.\n\n', 'Private ownership of TV satellite dishes is banned.\n\n', 'Private ownership is banned', 'ownership is banned'), (1, 'The claim is valid', "the country's third prime minister is head of government and is appointed by Lee Kuan Yew as the person most likely to command the confidence of a majority of Parliament.", "the country's third prime minister is head of government and is appointed by Lee Kuan Yew as the person most likely to command the confidence of a majority of Parliament."), (1, 'The claim is valid', 'The Singapore Art Museum hosting items from 50 countries.', 'The Singapore Art Museum celebrates exceptional art and design of objects for everyday life, hosting more than 1,000 items from 50 countries.'), (0, 'two statements are contradicting.', 'Self-censorship among journalists is said to be normal.', 'Self-censorship among journalists is said to be common.', 'Self-censorship is said normal', 'Self-censorship is said common'), (1, 'The claim is valid', 'Notable in this capacity has been the oil-refining industry.', 'Notable in this capacity has been the oil-refining industry.'), (0, "GPE does not match, refuted. Claim has Singapore, but fact has ['Singapore', 'Singapore']", 'In August 2017 the STB and the Economic Development Board (EDB) unveiled a unified brand, Singapore ??Passion Made Possible, to market Singapore internationally for tourism and business purposes.', 'In August 2017 the STB and the Economic Development Board (EDB) unveiled a unified brand, Singapore ??Passion Made Possible, to market Singapore internationally for tourism and business purposes.', 'Economic Development Board EDB unveiled internationally unified brand Singapore Passion Made Possible', 'Economic Development Board EDB unveiled unified brand Singapore Passion Made Possible to market Singapore'), (1, 'The claim is valid', 'An underlying principle has diplomatic relations with more than 180 countries.\n\n', 'An underlying principle has diplomatic relations with more than 180 sovereign states.\n\n'), (1, 'The claim is valid', 'These physical units reflect These physical units geologic foundations: the central hills are formed from granite rocks, the scarp lands from highly folded and faulted sedimentary rocks, and the eastern plateau from uncompacted sands and gravels.\n\n', 'These physical units reflect These physical units geologic foundations: the central hills are formed from granite rocks, the scarp lands from highly folded and faulted sedimentary rocks, and the eastern plateau from uncompacted sands and gravels.\n\n'), (1, 'The claim is valid', 'In response, Singapore has seen several significant political changes, such as the introduction of the Non-Constituency members of parliament in 1984 to allow up to three losing candidates from opposition parties to be appointed as MPs.', 'In response, Singapore has seen several significant political changes, such as the introduction of the Non-Constituency members of parliament in 1984 to allow up to three losing candidates from opposition parties to be appointed as MPs.'), (0, "GPE does not match, refuted. Claim has [['Singapore', 'Singapore'], 'Singapore'], but fact has Singapore", 'Singapore does not have a minimum wage.', 'Singapore does not have a minimum wage, believing that Singapore would lower Singapore competitiveness.', 'Singapore does not have a minimum wage.', 'Singapore does not have a minimum wage, believing that Singapore would lower Singapore competitiveness.'), (1, 'The claim is valid', 'Singapore ranked high on the factors of order and security .', 'Singapore ranked high on the factors of order and security (#1), absence of corruption (#3), regulatory enforcement (#3), civil justice (#5), and criminal justice (#6), but ranked significantly lower on factors of open government (#25), constraints on government powers (#27), and fundamental rights (#30).'), (1, 'The claim is valid', 'It has a symbolic, rather than functional purpose.', 'It has a symbolic, rather than functional purpose.'), (0, 'two statements are contradicting.', 'Danmaxi may be a transcription of Temasek (Tumasik).\n\n', 'Danmaxi may be a transcription of Temasek (Tumasik), alternatively, it may be a combination of the Malay Tanah meaning "land", and Chinese Xi meaning "tin", which was traded on the island.\n\n', 'Danmaxi may may transcription of Temasek', 'Danmaxi may may combination of Malay Tanah'), (1, 'The claim is valid', "It is seen as the guarantor of the country's independence.", "It is seen as the guarantor of the country's independence, translating into Singapore culture, involving all citizens in the country's defence."), (0, 'Fact does not relate to claim, cannot determine.', 'Parameswara founded the Sultanate of Malacca in Malacca.', 'Historical sources also indicate that around the end of the 14th century, Historical sources ruler Parameswara was attacked by either the Majapahit or the Siamese, forcing its ruler Parameswara to move to Malacca where its ruler Parameswara founded the Sultanate of Malacca.'), (1, 'The claim is valid', 'In London the English East India Company court of directors took no action.', 'In London the English East India Company court of directors, though the Dutch decided that Sir Stamford Raffles of the English East India Company had contravened instructions, took no action.'), (1, 'The claim is valid', 'Sir Stamford Raffles of the English East India Company offered to recognise Tengku Long as the rightful Sultan of Johor.', 'Sir Stamford Raffles of the English East India Company offered to recognise Tengku Long as the rightful Sultan of Johor, under the title of Sultan Hussein, as well as provide Sultan Hussein with a yearly payment of $5000 and another $3000 to the Temenggong; in return, Sultan Hussein would grant the British the right to establish a trading post on Singapore.'), (1, 'The claim is valid', 'However, according to official forecasts, water demand in Singapore is expected to double from 380 to 760 million US gallons (1.4 to 2.8 billion litres; 1.4 to 2.8 million cubic meters) per day between 2010 and 2060.', 'However, according to official forecasts, water demand in Singapore is expected to double from 380 to 760 million US gallons (1.4 to 2.8 billion litres; 1.4 to 2.8 million cubic meters) per day between 2010 and 2060.'), (0, 'two statements are contradicting.', 'In 1867, the Straits Settlements were separated from British India.', 'In 1867, the Straits Settlements were separated from British India, coming under the direct control of BritainBritain.', 'Straits Settlements were separated In 1867', 'Straits Settlements were separated coming under direct control of BritainBritain'), (0, 'two statements are contradicting.', 'Singapore is a major exporter of aquarium fish.', 'Singapore is a major exporter of both orchids and aquarium fish.', 'Singapore is major exporter of aquarium fish', 'Singapore is major exporter of orchids'), (1, 'The claim is valid', 'Education is valued in Singapore.', 'Education is highly valued in Singapore, and Education education system is elaborately structured.'), (0, 'two statements are contradicting.', 'To obtain a mandate for a merger, the PAP held a referendum on the merger.', 'To obtain a mandate for a merger, the PAP held a referendum on the merger.', 'PAP held referendum on merger', 'PAP obtain mandate for merger'), (1, 'The claim is valid', 'Sir Stamford Raffles of the English East India Company arrived in Singapore on 28 January 1819.', 'Sir Stamford Raffles of the English East India Company arrived in Singapore on 28 January 1819 and soon recognised the island as a natural choice for the new port.'), (1, 'The claim is valid', 'Rajendra Chola', 'Rajendra Chola'), (0, 'two statements are contradicting.', 'By 1860 the population had swelled to over 80,000, more than half being Chinese.', 'By 1860 the population had swelled to over 80,000, more than half being Chinese.', 'population had swelled half to over 80,000', 'population had swelled to over 80000 more'), (0, "GPE does not match, refuted. Claim has Singapore, but fact has ['Singapore', 'Penang']", 'Two years later Singapore, Penang, and Malacca (Melaka) were combined as the Straits Settlements to form an outlying residency of India.', 'Two years later Singapore, Penang, and Malacca (Melaka) were combined as the Straits Settlements to form an outlying residency of India.', 'Singapore were Two years later combined as Straits Settlements', 'Singapore Penang were Two years later combined as Straits Settlements'), (1, 'The claim is valid', 'Singapore and the United States share a long-standing close relationship.', 'Singapore and the United States share a long-standing close relationship, in particular in defence, the economy, health, and education.'), (1, 'The claim is valid', 'Although Multiracialism history stretches back millennia, Singapore Singapore was founded in 1819 by Sir Stamford Raffles of the English East India Company as a trading post of the British Empire.', 'Although Multiracialism history stretches back millennia, Singapore Singapore was founded in 1819 by Sir Stamford Raffles of the English East India Company as a trading post of the British Empire.'), (1, 'The claim is valid', 'Investments in the Indonesian island of Batam have been important.', 'Investments in the nearby Indonesian island of Batam have been important in this respect.'), (1, 'The claim is valid', 'Major imports are machinery and transport equipment and crude petroleum, while machinery and refined petroleum products are the major exports.', 'Major imports are machinery and transport equipment and crude petroleum, while machinery and refined petroleum products are the major exports.'), (1, 'The claim is valid', 'Lee Kuan Yew is head of state.', 'Lee Kuan Yew is head of state and exercises executive power on the advice of her ministers.'), (0, 'two statements are contradicting.', 'In October 1971, Britain pulled Britain military out of Singapore.', 'In addition, in October 1971, Britain pulled Britain military out of Singapore, leaving behind only a small British, Australian and New Zealand force as a token military presence.', 'Britain pulled Britain In October 1971', 'Britain military leaving behind only British Zealand force'), (0, 'No supported fact found for this claim, cannot determine.', 'Singapore medical services aim to generate US$3 billion in revenue.', 'Singapore medical services aim to serve at least one million foreign patients annually and generate US$3 billion in revenue.'), (1, 'The claim is valid', "The World Health Organisation ranks Singapore's healthcare system as 6th overall in the world.", "The World Health Organisation ranks Singapore's healthcare system as 6th overall in the world in The World Health Organisation World Health Report."), (1, 'The claim is valid', 'Although the historicity of the accounts as given in the Malay Annals is the subject of academic debates, the historicity of the accounts as given in the Malay Annals is the subject of academic debates is nevertheless known from various documents that Singapore in the 14th century, then known as Temasek, was a trading port under the influence of both the British Empire and the Siamese kingdoms, and was a part of the Indosphere.', 'Although the historicity of the accounts as given in the Malay Annals is the subject of academic debates, the historicity of the accounts as given in the Malay Annals is the subject of academic debates is nevertheless known from various documents that Singapore in the 14th century, then known as Temasek, was a trading port under the influence of both the British Empire and the Siamese kingdoms, and was a part of the Indosphere.'), (1, 'The claim is valid', 'Internal-security laws allows political dissidents to be held indefinitely without trial.', 'In addition, The PAP?�s often has suppressed and co-opted domestic opposition?�notably through internal-security laws that allow political dissidents to be held indefinitely without trial?�and The PAP?�s has promoted a national paternalistic ideology through a variety of laws and corporate institutions.'), (0, "GPE does not match, refuted. Claim has Singapore, but fact has ['Singapore', 'Singapore']", 'Singapore attracts a large amount of foreign investment.', 'Singapore attracts a large amount of foreign investment as a result of Singapore location, skilled workforce, low tax rates, advanced infrastructure and zero-tolerance against corruption.', 'Singapore attracts amount of investment', 'Singapore attracts amount as result of Singapore location'), (1, 'The claim is valid', 'The city of Singapore is situated in the southern of the island.', 'The city of Singapore is situated in the southern portion of the island.'), (0, 'two statements are contradicting.', "The combined area of which has increased by 25% since The country's independence.", "The country's territory is composed of one main island, 63 satellite islands and islets, and one outlying islet, the combined area of which has increased by 25% since The country's independence as a result of extensive land reclamation projects.", 'combined area has increased by 25', 'combined area has increased countrys independence as result of land reclamation projects'), (0, 'two statements are contradicting.', 'Only a tiny fraction of the land area is classified as agricultural.', 'Only a tiny fraction of the land area is classified as agricultural, and production contributes a negligible amount to Singapore?�s economy.', 'tiny fraction is classified', 'classified negligible amount to Singapore'), (1, 'The claim is valid', "Pedra Branca is Singapore's easternmost point.\n\n", "Pedra Branca is Singapore's easternmost point.\n\n"), (0, 'two statements are contradicting.', 'Presidential elections may be declared "reserved" for a racial community.', 'Presidential elections may be declared "reserved" for a racial community if no one from that ethnic group has been elected to the presidency in the five most recent terms.', 'elections may may declared', 'Presidential elections reserved has elected to presidency in five most recent terms'), (1, 'The claim is valid', 'In hawker centres, cultural diffusion is exemplified by traditionally Malay hawker stalls also selling Tamil food.', 'In hawker centres, cultural diffusion is exemplified by traditionally Malay hawker stalls also selling Tamil food.'), (1, 'The claim is valid', 'Equinix (332 participants) and also Equinix (332 participants)', 'Equinix (332 participants) and also Equinix (332 participants)'), (1, 'The claim is valid', 'Education takes place in three stages.', 'Education takes place in three stages: primary, secondary, and pre-university education.'), (1, 'The claim is valid', 'Singapore is the largest port in Southeast Asia.', 'Singapore is the largest port in Southeast Asia and one of the busiest in the world.'), (0, 'two statements are contradicting.', 'Singapore gained self-governance in 1959.', 'Singapore gained self-governance in 1959, and in 1963 became part of the new federation of Malaysia, alongside Malaya, North Borneo, and Sarawak.', 'Singapore gained self-governance in 1959', 'Singapore became part of federation of Malaysia'), (0, "GPE does not match, refuted. Claim has ['Singapore', 'Malaysia'], but fact has Malaysia", 'In an attempt to foster additional trade, Singapore has become a joint-venture partner in numerous projects with Malaysia.', 'In an attempt to foster additional trade, Singapore has become a joint-venture partner in numerous projects with Malaysia and Indonesia.', 'Singapore joint-venture partner in projects with Malaysia', 'jointventure partner is in numerous projects with Malaysia'), (1, 'The claim is valid', 'Members of Parliament (MPs) are chosen to serve for a term lasting up to 5 years.', 'Members of Parliament (MPs) are chosen to serve for a term lasting up to five years.'), (1, 'The claim is valid', 'The Defence Science and Technology Agency is responsible for procuring resources.', 'The Defence Science and Technology Agency is responsible for procuring resources for the military.'), (1, 'The claim is valid', 'The government Monetary Authority of Singapore oversees all shipping activity and operates a number of terminals on the island.', 'The government?�s Monetary Authority of Singapore oversees all shipping activity and operates a number of terminals on the island.'), (1, 'The claim is valid', 'A Sharīʿah court has jurisdiction in matters of Islamic law.', 'A Sharīʿah court has jurisdiction in matters of Islamic law.'), (1, 'The claim is valid', 'Singapore has one of the highest income inequalities.', 'Singapore also has one of the highest income inequalities among developed countries.'), (1, 'The claim is valid', 'looting and revenge-killing were widespread.', 'After the Japanese surrender to the Allies on 15 August 1945, Singapore fell into a brief state of violence and disorder; looting and revenge-killing were widespread.'), (1, 'The claim is valid', 'the large Singapore Naval Base was defended by heavy 15-inch naval guns stationed at Fort Siloso, Fort Canning and Labrador, as well as a Royal Air Force airfield at Tengah Air Base.', 'the large Singapore Naval Base was defended by heavy 15-inch naval guns stationed at Fort Siloso, Fort Canning and Labrador, as well as a Royal Air Force airfield at Tengah Air Base.'), (1, 'The claim is valid', 'Since 2009, the Republic of Singapore Navy (RSN) has deployed ships to the Gulf of Aden to aid in counter piracy efforts.', 'Since 2009, the Republic of Singapore Navy (RSN) has deployed ships to the Gulf of Aden to aid in counter piracy efforts as part of Task Force 151.'), (1, 'The claim is valid', 'Singapore serving some of the busiest sea and air trade routes.', 'Singapore is a major international transport hub in Asia, serving some of the busiest sea and air trade routes.'), (1, 'The claim is valid', 'the main island is separated from Peninsular Malaysia', 'the main island is separated from Peninsular Malaysia to the north by Johor Strait, a narrow channel crossed by a road and rail causeway that is more than half a mile long.'), (1, 'The claim is valid', "Singapore's best known global companies include Singapore Airlines", "Singapore's best known global companies include Singapore Airlines, Changi Airport, and the Port of Singapore, all of which are among the most-awarded in The nation's best known global companies respective fields."), (1, 'The claim is valid', 'The soils of eastern Singapore are extremely infertile.', 'The soils of eastern Singapore are extremely infertile.'), (1, 'The claim is valid', 'None of those three major communities is homogeneous.', 'None of those three major communities is homogeneous.'), (1, 'The claim is valid', 'In Singapore, street food has long been associated with hawker centres.', 'In Singapore, street food has long been associated with hawker centres with communal seating areas.'), (1, 'The claim is valid', 'Singapore seceded to become an independent state on August 9, 1965.\n', 'Once a British colony and now a member of the Commonwealth, Singapore first joined the Federation of Malaysia on the Federation of Malaysia formation in 1963 but seceded to become an independent state on August 9, 1965.\n'), (1, 'The claim is valid', "Hainanese chicken rice is considered Singapore's national dish.\n\n", "Hainanese chicken rice, based on the Hainanese dish Wenchang chicken, is considered Singapore's national dish.\n\n"), (1, 'The claim is valid', "The large number of immigrants has kept Singapore's population from decreasing.\n\n", "The large number of immigrants has kept Singapore's population from declining.\n\n"), (0, 'two statements are contradicting.', ' Rainfall dropping to a monthly low of less than 7 inches in July.', 'Conversely, the period of the least amount of rainfall and the lightest winds is during the southwest monsoon (May?�September), with rainfall dropping to a monthly low of less than 7 inches in July.', 'Rainfall dropping to monthly low', 'amount Conversely period of is southwest monsoon May September with rainfall dropping to low'), (0, 'two statements are contradicting.', 'Presidential elections are held using first-past-the-post voting.', 'Presidential elections are held using first-past-the-post voting.', 'elections using first-past-the-post voting', 'elections using firstpastthepost voting'), (0, "DATE does not match, refuted. Claim has October, but fact has ['late', 'October']", 'The earliest the sun rises and sets is in late October and early November when the sun rises at 6:46 am and sets at 6:50 pm.\n\n', 'The earliest the sun rises and sets is in late October and early November when the sun rises at 6:46 am and sets at 6:50 pm.\n\n', 'earliest is in October', 'earliest is in late October'), (1, 'The claim is valid', 'The small size of the population has also affected the way SAF SAF has been designed.\n\n', 'The small size of the population has also affected the way SAF SAF has been designed, with a small active force but a large number of reserves.\n\n'), (1, 'The claim is valid', 'Lee Kuan Yew is directly elected by popular vote.', 'Lee Kuan Yew is directly elected by popular vote for a renewable six-year term.'), (0, "DATE does not match, refuted. Claim has [[[['the', 'year'], 'November'], 'to'], 'February'], but fact has [[[[[['a', 'wetter'], 'monsoon'], 'season'], 'November'], 'to'], 'February']", 'there is a wetter monsoon season from November to February.\n\n', 'While temperature does not vary greatly throughout the year, there is a wetter monsoon season from November to February.\n\n', 'there is a wetter monsoon season from November to February.\n\n', 'While temperature does not vary greatly throughout the year, there is a wetter monsoon season from November to February.\n\n'), (0, 'two statements are contradicting.', 'Freedom House ranks Singapore as "partly free" in Freedom House Freedom in the Report', 'Freedom House ranks Singapore as "partly free" in Freedom House Freedom in the Report, and The Economist Intelligence Unit ranks Singapore as a "flawed democracy", the second best rank of four, in The Economist Intelligence Unit "Democracy Index".', 'Freedom House ranks Singapore as free', 'Economist Intelligence Unit ranks Singapore as flawed democracy'), (1, 'The claim is valid', 'the Kingdom of Singapura was founded on the island by Sang Nila Utama.', 'In 1299, according to the Malay Annals, the Kingdom of Singapura was founded on the island by Sang Nila Utama.'), (1, 'The claim is valid', 'Live-in foreign domestic workers are quite common in Singapore\n\n', 'Live-in foreign domestic workers are quite common in Singapore, with about 224,500 foreign domestic workers there, as of December 2013.\n\n'), (1, 'The claim is valid', 'Considered too small to provide effective security for the country, the development of the country military forces became a priority.', 'Considered too small to provide effective security for the country, the development of the country military forces became a priority.'), (0, 'two statements are contradicting.', 'The geographic restrictions of Singapore mean that SAF must plan to fully repulse an attack', 'The geographic restrictions of Singapore mean that SAF must plan to fully repulse an attack, as they cannot fall back and re-group.', 'SAF fully repulse attack', 'SAF repulse attack'), (0, 'two statements are contradicting.', 'Temasek fell into decay and was supplanted by Malacca (now Melaka).', 'At the end of the 14th century, Temasek fell into decay and was supplanted by Malacca (now Melaka).', 'Temasek was supplanted by now Melaka', 'Temasek was supplanted by Melaka'), (1, 'The claim is valid', 'Tourism has become increasingly important to Singapore economy.', 'Tourism has become increasingly important to Singapore economy.'), (1, 'The claim is valid', 'During the 1950s, Chinese communists, with strong ties to the trade unions and Chinese schools, waged a guerrilla war against the government, leading to the Malayan Emergency.', 'During the 1950s, Chinese communists, with strong ties to the trade unions and Chinese schools, waged a guerrilla war against the government, leading to the Malayan Emergency.'), (0, "GPE does not match, refuted. Claim has ['China', 'Singapore'], but fact has China", " China has been Singapore's largest trading partner since 2013", "In addition, China has been Singapore's largest trading partner since 2013, after surpassing Malaysia.", "China has has Singapore 's trading partner", 'China has In addition has largest trading partner since 2013'), (1, 'The claim is valid', 'In Javanese inscriptions and Chinese records dating to the end of the 14th century, the more-common name of the island is Tumasik, or Temasek, from the Javanese word tasek (?�sea??.', 'In Javanese inscriptions and Chinese records dating to the end of the 14th century, the more-common name of the island is Tumasik, or Temasek, from the Javanese word tasek (?�sea??.'), (1, 'The claim is valid', 'The Straits Times reported that Indonesia had decided to create tax havens on two islands near Singapore to bring Indonesian capital back into the tax base.', 'In August 2016, The Straits Times reported that Indonesia had decided to create tax havens on two islands near Singapore to bring Indonesian capital back into the tax base.'), (1, 'The claim is valid', 'Temasek was christened Singapura by Sang Nila Utama, a 13th-century Srivijayan Raja from Palembang encountered the beast.', 'Seeing this as an omen, The semi-historical Malay Annals state that Temasek was christened Singapura by Sang Nila Utama, a 13th-century Srivijayan Raja from Palembang established the town of Singapura where The semi-historical Malay Annals state that Temasek was christened Singapura by Sang Nila Utama, a 13th-century Srivijayan Raja from Palembang encountered the beast.'), (0, "CARDINAL does not match, refuted. Claim has ['6', '28000'], but fact has 6", 'There are six taxi companies', 'There are six taxi companies, who together put out over 28,000 taxis on the road.', 'There are six taxi companies', 'There are six taxi companies, who together put out over 28,000 taxis on the road.'), (1, 'The claim is valid', 'Jawi is considered an ethnic script for use on Singaporean identity cards.\n\n', 'Jawi is considered an ethnic script for use on Singaporean identity cards.\n\n'), (0, 'two statements are contradicting.', 'In 1613, Portuguese raiders burned down the main settlement on Fort Canning', 'In 1613, Portuguese raiders burned down the main settlement on Fort Canning, and the island faded into obscurity for the next two centuries.', 'Portuguese raiders burned down settlement', '1613 Portuguese raiders faded for next two centuries'), (0, 'two statements are contradicting.', 'the Singapore Symphony Orchestra (SSO) instituted in 1979.', 'Western classical music plays a significant role in the cultural life in Singapore, with the Singapore Symphony Orchestra (SSO) instituted in 1979.', 'Singapore Symphony Orchestra instituted in 1979', 'Western classical music plays role with Singapore Symphony Orchestra SSO instituted'), (1, 'The claim is valid', 'The Singaporean military, arguably the most technologically advanced in Southeast Asia, consists of the army, navy, and the air force.', 'The Singaporean military, arguably the most technologically advanced in Southeast Asia, consists of the army, navy, and the air force.'), (0, 'No supported fact found for this claim, cannot determine.', 'The failure of Britain changed the Japanese image in the eyes of Singaporeans.', 'The failure of Britain to successfully defend Britain colony against the Japanese changed the Japanese image in the eyes of Singaporeans.'), (1, 'The claim is valid', 'Adult obesity is below 10%.', 'Adult obesity is below 10%.')]
# for claim, fact_stmt in zip(test_set['sentences'][100:], test_set['sentences_original'][100:]):
#     result = check_result_given_fact(claim, fact_stmt)
#     pred_given.append(result)
# len(pred_given)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
#
# y_pred_given = []
# for item in pred_given:
#     y_pred_given.append(item[0])
#
# cm = confusion_matrix(test_set['target'], y_pred_given, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=["false", 'true'])
# disp.plot()
#
# result_df = pd.DataFrame(pred_given, columns =['label', 'reason','claim','fact','atomic_claim','atomic_fact'])
# result_df.to_csv('result_csv.csv')
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

#Usage1: "python main.py" for comparing two statements
#Usage2: "python main.py -m kb" for checking against the knowledge base
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=False)
    parser.add_argument("-d", "--debug", required=False)
    args = parser.parse_args()
    mode = args.mode
    debug = args.debug
    print("Debug mode is",debug)
    if mode =="kb":
        print("")
        print("")
        print(colored('=========================================','green'))
        print(colored("Welcome to the fact checking system about Singapore",'green'))
        loop = True
        while loop:
            claim = input(colored("Enter a claim about Singapore to check: ",'green'))
            print(colored("We are comparing the fact and the claim, please wait...",'green'))
            with HiddenPrints():
                result = fact_check_result_naive(claim, df)
            print(colored(("RESULT", result[1]),'green'))
            print(result)

            again = input(colored("Do you want to try again?",'green'))
            if again.startswith("y") or again.startswith("Y"):
                loop = True
            else:
                loop = False
    else:
        print("")
        print("")
        print(colored('=========================================','green'))
        print(colored("Welcome to the fact checking system.",'green'))
        loop = True
        while loop:
            fact = input(colored("Enter a fact: ",'green'))
            claim = input(colored("Enter a claim to compare: ",'green'))
            print(colored("We are comparing the fact and the claim, please wait...",'green'))
            if debug == "True":
                result = check_result_given_fact(claim, fact)
            else:
                with HiddenPrints():
                    result = check_result_given_fact(claim, fact)
            print(colored(("RESULT", result[1]),'green'))
            print(colored(result,'green'))

            again = input(colored("Do you want to try again?",'green'))
            if again.startswith("y") or again.startswith("Y"):
                loop = True
            else:
                loop = False



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

check_semantic_similarity('This is a lady','This is a girl')
test_data_generator('This is a lady','This is a girl').__getitem__(0)