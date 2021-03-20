# -- coding: utf-8 --

import os
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials

SUBSCRIPTION_KEY_ENV_NAME = "bc20ced3c3014badbf34d1799e28f2a2"
#SUBSCRIPTION_KEY_ENV_NAME = '6f10bc0d-296b-44c3-9a6c-d55c966dfc33'
TEXTANALYTICS_LOCATION = os.environ.get(
    "https://text-analysis-smartg.cognitiveservices.azure.com/", "westus2")


def language_extraction(subscription_key):
    """Language extraction.
    This example detects the language of several strings. 
    """
    credentials = CognitiveServicesCredentials(subscription_key)
    text_analytics_url = "https://{}.api.cognitive.microsoft.com".format(
        TEXTANALYTICS_LOCATION)
    text_analytics = TextAnalyticsClient(
        endpoint=text_analytics_url, credentials=credentials)

    try:
        documents = [
            {'id': '1', 'text': 'This is a document written in English.'},
            {'id': '2', 'text': 'Este es un document escrito en Español.'},
            {'id': '3', 'text': '这是一个用中文写的文件'}
        ]
        response = text_analytics.detect_language(documents=documents)

        for document in response.documents:
            print("Document Id: ", document.id, ", Language: ",
                  document.detected_languages[0].name)

    except Exception as err:
        print("Encountered exception. {}".format(err))


def key_phrases(subscription_key,text):
    """Key-phrases.
    Returns the key talking points in several text examples.
    """
    credentials = CognitiveServicesCredentials(subscription_key)
    text_analytics_url = "https://{}.api.cognitive.microsoft.com".format(
        TEXTANALYTICS_LOCATION)
    text_analytics = TextAnalyticsClient(
        endpoint=text_analytics_url, credentials=credentials)

    try:
        documents = [
            {"id": "1", "language": "en", "text": text}
        ]

        for document in documents:
            '''print(
                #"Asking key-phrases on '{}' (id: {})".format(document['text'], document['id']))'''
        response = text_analytics.key_phrases(documents=documents)

        for document in response.documents:
            #print("Document Id: ", document.id)
            #print("\tKey Phrases:")
            x=''
            for phrase in document.key_phrases:
                x+=phrase+'\n'
                #print("\t\t", phrase)
        
        ret_list = [documents[0]['text'],x]
        
        return ret_list

    except Exception as err:
        print("Encountered exception. {}".format(err))


def sentiment(subscription_key,text):
    """Sentiment.
    Scores close to 1 indicate positive sentiment, while scores close to 0 indicate negative sentiment.
    """
    credentials = CognitiveServicesCredentials(subscription_key)
    text_analytics_url = "https://{}.api.cognitive.microsoft.com".format(
        TEXTANALYTICS_LOCATION)
    text_analytics = TextAnalyticsClient(
        endpoint=text_analytics_url, credentials=credentials)

    try:
        documents = [
            {"id": "1", "language": "en", "text": text},
        ]

        response = text_analytics.sentiment(documents=documents)
        for document in response.documents:
            print("Document Id: ", document.id, ", Sentiment Score: ",
                  "{:.2f}".format(document.score))

    except Exception as err:
        print("Encountered exception. {}".format(err))


def entity_extraction(subscription_key,text):
    """EntityExtraction.
    Extracts the entities from sentences and prints out their properties.
    """
    credentials = CognitiveServicesCredentials(subscription_key)
    text_analytics_url = "https://{}.api.cognitive.microsoft.com".format(
        TEXTANALYTICS_LOCATION)
    text_analytics = TextAnalyticsClient(
        endpoint=text_analytics_url, credentials=credentials)

    try:
        documents = [
            {"id": "1", "language": "en", "text": text},
        ]
        response = text_analytics.entities(documents=documents)

        for document in response.documents:
            print("Document Id: ", document.id)
            print("\tKey Entities:")
            #return document.entities 
            for entity in document.entities:
                print("\t\t", "NAME: ", entity.name, "\tType: ",
                      entity.type, "\tSub-type: ", entity.sub_type)
                ml = [entity.name, entity.type, entity.sub_type]
                return ml
                #for match in entity.matches:
                        #print("\t\t\tOffset: ", match.offset, "\tLength: ", match.length, "\tScore: ","{:.2f}".format(match.entity_type_score))'''

    except Exception as err:
        print("Encountered exception. {}".format(err))
        return 0


if __name__ == "__main__":
    import sys
    import os.path

    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

    keys = key_phrases(SUBSCRIPTION_KEY_ENV_NAME,"hey akshat")
    ent = entity_extraction(SUBSCRIPTION_KEY_ENV_NAME,"take me to Sydney Town Hall")
    #print(keys)
    print(ent)
    #print('input text: ',keys[0])
    #print('phrases: ',keys[1])
    print(sentiment(SUBSCRIPTION_KEY_ENV_NAME,"THE PRODUCT IS TRASH"))