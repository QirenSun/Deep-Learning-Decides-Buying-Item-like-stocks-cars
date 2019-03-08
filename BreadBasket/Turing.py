
import json
import urllib.request
def chat(text_input):       
    if text_input=='q':        
        return
    else:
            
        req = {
        "perception":
        {
            "inputText":
            {
                "text": text_input
            },
        
            "selfInfo":
            {
                "location":
                {
                    "city": "",
                    "province": "",
                    "street": ''
                }
            }
        },
        
        "userInfo": 
        {
            "apiKey": '86c95f8541d94cfe83089918b3cab9d1',
            "userId": '411384'
        }
        }
        
        req = json.dumps(req).encode('utf8')
        # print(req)
        api_url='http://openapi.tuling123.com/openapi/api/v2'
        http_post = urllib.request.Request(api_url, data=req, headers={'content-type': 'application/json'})
        response = urllib.request.urlopen(http_post)
        response_str = response.read().decode('utf8')
        # print(response_str)
        response_dic = json.loads(response_str)
        # print(response_dic)
        
        #intent_code = response_dic['intent']['code']
        results_text = response_dic['results'][0]['values']['text']
        #print('Turing的回答：')
        #print('code：' + str(intent_code))
        print('Turing：' + results_text)
    return(chat(input('我：')))


chat(input('我: '))



