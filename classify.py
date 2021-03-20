import ibm_watson
def cla(review):
    assistant = ibm_watson.AssistantV1(
    version='2019-02-28',
    iam_apikey='u1N9ThXmpZUk_-1_F1AaAw-11BbBXFtCbonmmerHbnFI',
    url='https://gateway-wdc.watsonplatform.net/assistant/api/'
    )

    response = assistant.message(
        workspace_id='7cb1c0fc-6e91-4b63-9e93-8a30028bd58e',
        input={
            'text': review 
        }
    ).get_result()


    a=response
    b=a['intents']
    print(b)
if __name__ == "__main__":
    cla('add two numbers')