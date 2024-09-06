import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords



# AФункция удаления стоп слов из текстов
def remove_stop_words(sentence, stop):
    clean_sentence = ' '.join([token for token in sentence if token not in stop])
    return(clean_sentence)


# Функция получения эмбедингов
def emb(text, model):
    '''
    text - преобразуемый текст в виде списка
    model - модель SentenceTransformer для получения эмбедингов
    '''
    database_vectors_emb = []
    # Получение эмбедингов
    for name  in text:
        text_embedding = model.encode(name, convert_to_tensor=False)
        database_vectors_emb.append(text_embedding)
    return database_vectors_emb


def get_result(text: pd.Series) -> pd.Series:

    # Загрузка модели получения эмбедингов
    model_emb = joblib.load('../models/rubert-tiny2.pkl')
    # Загрузка модели классификации
    class_model = joblib.load('../models/class_model.pkl')
    # Вызов стоп-слов из библиотеки "nltk"
    stop_words = stopwords.words('russian')

    # Перевод текстов в нижний регистр
    text= text.str.lower()
    # Токенизация текста с помощью модуля "nltk"
    text = text.apply(lambda x: word_tokenize(x, language='russian'))
    # Удаление стоп-слов из текстов
    text = text.apply(remove_stop_words, stop=stop_words)
    # Получение эмбедингов
    text_emb = emb(text, model_emb)
    # Получение предсказаний модели
    class_predicted = class_model.predict(text_emb)

    return pd.Series(class_predicted, name='class_predicted')


if __name__ == '__main__':

    # Тестирование модели
    text_sample = pd.Series(['кассир нервная !',
                             'очень грубый продавец',
                             'бесконечно долго ждал очереди'])

    class_predicted = get_result(text_sample)
    print(class_predicted)

