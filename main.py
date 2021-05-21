# Importing libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from PIL import Image

# Set title.
st.title("Asunnon hinta-arvio")

st.write("""
### Tampere, Suomi
""")

# Reading data to dataframes.
data = pd.read_csv("cleaned_data.csv")
alldata = pd.read_csv("all_data.csv")
df = alldata[0:0]
df = df.drop(['Hinta'], axis=1)

# Creating input boxes and select boxes.
s = st.sidebar.text_input("Neliöt", 0)
y = st.sidebar.text_input("Rakennusvuosi", 0)

c_list = data['Kaupunginosa'].tolist()
options_city = list(dict.fromkeys(c_list))
area_name = st.sidebar.selectbox("Kaupunginosa", options_city)

t_list = data['Tyyppi'].tolist()
options_type = list(dict.fromkeys(t_list))
type_name = st.sidebar.selectbox("Asunnon tyyppi", options_type)

# Adding new row to dataframe based on users input.
new_row = {'Neliöt': s, 'Rakennusvuosi': y, area_name: 1, type_name: 1}
df = df.append(new_row, ignore_index=True)
df = df.fillna(0)

# Setting image on page.
image = Image.open('keskustori_ilmasta.jpg')
st.image(image)

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(
    alldata.drop(['Hinta'], axis=1),
    alldata.Hinta, test_size=0.2, random_state=20)

# Fitting data to model.
def model_pipeline(model, param_grid, scoring):
    Tuned_Model = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring=scoring, cv=3)

    Tuned_Model.fit(X_train, y_train)
    return Tuned_Model

param_grid = {
    "n_estimators": [500],
    "max_features": [0.1],
}
model2 = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True)
Tuned_Model2 = model_pipeline(model2, param_grid, "neg_median_absolute_error")
best2 = Tuned_Model2.best_estimator_

# Predicting price for user input.
y_pred = Tuned_Model2.predict(df)
hinta = int(y_pred)
hinta = str(hinta)
s = 'Neliöitä: ' + s + 'm^2' + ', Rakennusvuosi: ' + y + ', Kaupunginosa: ' + area_name + ', Asunnon tyyppi: ' + type_name


h_ylaraja = int(hinta)*1.1
h_alaraja = int(hinta)*0.9
h_ylaraja = int(h_ylaraja)
h_alaraja = int(h_alaraja)
h_ylaraja = str(h_ylaraja)
h_alaraja = str(h_alaraja)
arvio = 'Asunnon hinta sijoittuu todennäköisesti välille ' + h_alaraja + ' - ' + h_ylaraja + ' €'

# Printing predicted price for user.
if st.sidebar.button('Laske hinta-arvio'):
    h = str("Asuntosi arvioitu hinta on noin " + hinta + " €")
    st.subheader(h)
    arvio


