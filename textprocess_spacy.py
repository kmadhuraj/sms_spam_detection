import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

document_spam = nlp(spam_string)
document_ham = nlp(ham_string)

displacy.render(document_spam, style = 'ent', jupyter = True)

displacy.render(document_ham, style = 'ent', jupyter = True)

person_list = [] 
for entity in document_ham.ents:
    if entity.label_ == 'PERSON':
        person_list.append(entity.text)
        
person_list[0: 10]