import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://upload.wikimedia.org/wikipedia/commons/7/71/Black.png");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)


##import data
data = pd.read_csv("hi.csv")
st.title('📢:blue[ทำนายเงินเดือนของคุณ]')
st.write(":green[นี่เป็นแอปพลิเคชั่นประเมินเงินเดือนของคุณ]")
st.write(":green[มาดูกันว่าเงินเดือนของคุณจะเป็นอย่างไร]")
#splitting  data
data['experience'].fillna(0, inplace=True)
data['test_score'].fillna(data['test_score'].mean(),inplace=True)
X = data.iloc[:,:3]

#Converting words to integer values
def convert_to_int(world):
    world_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,
                  'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return world_dict[world]
X ['experience'] = X['experience'].apply(lambda x: convert_to_int(x))
y = data.iloc[:,-1]

#input the numbers
Experience = st.slider(":red[ประสบการณ์ของคุณ ?]",int(X.experience.min()),int(X.experience.max()),int(X.experience.mean()))
Test_score = st.slider(":red[คะแนนสอบของคุณ ?]",int(data.test_score.min()),int(data.test_score.max()),int(data.test_score.mean()))
Interview_score = st.slider(":red[คะแนนสัมภาษณ์ของคุณ ?]",int(data.interview_score.min()),int(data.interview_score.max()),int(data.interview_score.mean()))

#model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict([[Experience,Test_score,Interview_score]])[0]
# Save model
joblib.dump(model,'Predict_M.joblib')
#load model
lr = joblib.load('Predict_M.joblib')
predict = lr.predict(X)

#checking prediction
if st.button ("ทำนายเงินเดือน !"):
    st.header("🎉:red[เงินเดือนของคุณอยู่ที่]  {} :red[บาท]".format(int(predictions)))
