import os
import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from flask import Flask, render_template
import googleapiclient.discovery

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

from flask_bootstrap import Bootstrap5
bootstrap5 = Bootstrap5(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField 
from wtforms.validators import DataRequired
from sklearn.model_selection import train_test_split
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

class LabForm(FlaskForm):
    longtitude = StringField('longtitude(1-7)', validators=[DataRequired()]) 
    latitude = StringField('latitude(1-7)', validators=[DataRequired()]) 
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()]) 
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()]) 
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()]) 
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index') 
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form=LabForm()
    if form.validate_on_submit():
        # get the dorm data for the patient data and put into a form for the 
        X_test = np.array([[float(form.longtitude.data),
                            float(form.latitude.data),
                            str(form.month.data),
                            str(form.day.data),
                            float(form.avg_temp.data),
                            float(form.max_temp.data),
                            float(form.max_wind_speed.data),
                            float(form.avg_wind.data)]])
        
        # in order to make a prediction,
        # we must scale the data using the same scale as the one used to make model
        #get the data for fires data.

        X_test = pd.DataFrame(X_test, columns=[ 'latitude', 'longitude', 'month', 'day',
                                               'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'])
        print(X_test.shape)
        print(X_test)
        
        # get the data for the diabetes data.   
        data = pd.read_csv('./sanbul-5.csv', sep=',')

        # 훈련 데이터셋 정의
        fires_train = data  # 전체 데이터를 사용하는 경우

        # 수치형 특성 선택
        fires_train_num = fires_train[['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']]

        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])

        num_attribs = list(fires_train_num.columns)
        cat_attribs = ['month', 'day']

        full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs),
            ('cat', OneHotEncoder(), cat_attribs)
        ])
     
        full_pipeline.fit(fires_train)
        X_test = full_pipeline.transform(X_test)

        # create the resource to the model web api on GCP
        
        project_id = 'gen-lang-client-0924727192' 
        model_id = "sanbul_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="gen-lang-client-0924727192-e15be119c057.json"
        model_path = "projects/{}/models/{}".format(project_id, model_id) 
        model_path += "/versions/v0001/"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        # format the data as a json to send to the web api
        input_data_json={"signature_name": "serving_default", "instances": X_test.tolist()}
        
        # make the prediction
        request = ml_resource.predict(name=model_path, body=input_data_json) 
        response = request.execute()
        print("\nresponse: \n", response)

        if "error" in response:
            raise RuntimeError(response["error"])
        
        predD=np.array([pred['dense_3'][0] for pred in response["predictions"]])

        # 백분율 값을 면적으로 변환
        area_result = convert_to_area_percentage(predD[0])
        
        return render_template('result.html', res=area_result) 
    return render_template('prediction.html', form=form)

def convert_to_area_percentage(percentage):
    # 백분율 값을 면적으로 변환 (현재 백분율 값은 0과 1 사이의 값이므로 100을 곱해줍니다.)
    area = percentage * 100
    return area

if __name__ =='__main__':
    app.run()
