from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions

def get_senti(text):

    authenticator = IAMAuthenticator('{apikey}')
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url('https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

    response = natural_language_understanding.analyze(
        text=text,
        features=Features(sentiment=SentimentOptions())).get_result()
    b=response['sentiment']['document']
    return b

    #print(json.dumps(response, indent=2))

if __name__ == "__main__":
    senti = get_senti('this product is amazing')
    print(senti)
