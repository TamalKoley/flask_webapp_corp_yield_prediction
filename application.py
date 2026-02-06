from flask import Flask,request,render_template;
import pickle;

application=Flask(__name__);
app=application
reg_model=pickle.load(open('models/ridgecv_model.pkl','rb'));
#scalar_model=pickle.load(open('models/scalar_model.pkl','rb'));
poly_model=pickle.load(open('models/poly_model.pkl','rb'));

@app.route("/",methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html');
    else:
        fertilizer=float(request.form.get('fertilizer'))
        water=float(request.form.get('water'));
        #scaled_val=scalar_model.transform([[fertilizer,water]])
        poly_scaled_val=poly_model.transform([[fertilizer,water]])
        pred_val=reg_model.predict(poly_scaled_val)
        return render_template('index.html',results=pred_val[0])

        

if __name__=='__main__':
    app.run(host="0.0.0.0")
