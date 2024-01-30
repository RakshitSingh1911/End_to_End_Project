from Insurance.pipeline.single_prediction_pipeline import CustomData,PredictPipeline


from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data= CustomData(

            age = float(request.form.get('Age')),
            sex = request.form.get('Sex'),
            bmi = float(request.form.get('Bmi')),
            children = float(request.form.get('children')),
            smoker = request.form.get('Smoker'),
            region = request.form.get('Region'),
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.PREDICT(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
