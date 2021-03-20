from flask import Flask, render_template, request
import webbrowser, json
import runner as r
import normalize as n
import time


app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def main():
    return render_template('index.html')

    
@app.route('/url',methods=['POST','GET'])
def url():
    if request.method == 'POST':
        resp_json = request.get_json()
        f =  resp_json['text'] #link
        print(f)
        time.sleep(2)
        r.parse2(f)
        new_rating = n.get_nr()

        return json.dumps({"response": new_rating}), 200

if __name__ == '__main__':
    #webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=False)
