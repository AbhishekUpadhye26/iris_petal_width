from flask import Flask,render_template,request
import pickle as pkl

app = Flask(__name__)
model = pkl.load(open("iris_reg_model.pkl","rb"))
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediction",methods=["POST","GET"])
def iris_new():

    var_sl = float(request.form.get('sepal_length'))
    var_sw = float(request.form.get("sepal_width"))
    var_pl = float(request.form.get("petal_length"))
    var_species = int(request.form.get("species"))

    # print(f"{var_sl},{var_sw},{var_pl},{var_species}")
    result = model.predict([[var_sl,var_sw,var_pl,var_species]])

    print(result[0])
    return "PLACED"



if __name__ =="__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)