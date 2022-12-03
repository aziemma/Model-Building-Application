import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#matplotlib.use('Agg')

st.title("Interactive Data Application by Emmanuel Azi-love")

def main():
	activities=['EDA','Visualization','Model','Background']
	option=st.sidebar.selectbox("Select Option",activities)

	if option == 'EDA':
		st.subheader("Exploratory Data Analysis")
		data=st.file_uploader("Upload dataset", type=['csv','xlsx','txt','json'])

		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			st.success("Data Successfully uploaded")

			if st.checkbox("Display Shape"):
				st.write(df.shape)
			if st.checkbox("Display Columns"):
				st.write(df.columns)
			if st.checkbox("Select Multiple columns"):
				selected_columns=st.multiselect("select prefered columns:",df.columns)
				df1=df[selected_columns]
				st.write(df1)
			if st.checkbox("Display Summary of selected columns"):
				st.write(df1.describe().T)
			if st.checkbox("Check for null values"):
				st.write(df.isna().sum())
			if st.checkbox("Display data types"):
				st.write(df.dtypes)
			if st.checkbox("Display correlations of selected columns"):
				st.write(df1.corr())

	elif option == 'Visualization':
		st.subheader("Data Visualization")
		data=st.file_uploader("Upload dataset for visualization",type=['csv','xlsx','txt','json'])

		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			st.success("Data Successfully uploaded!")

			if st.checkbox("Select Multiple columns to plot"):
				selected_columns=st.multiselect("Select prefered columns:",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox("Heat Map Display"):
				st.write(sns.heatmap(df1.corr(),square=True,vmax=1,annot=True,cmap='plasma'))
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.pyplot()
			if st.checkbox("Display pairplot"):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()
			if st.checkbox("Display piechart"):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox('select column to display piechart:',all_columns)
				pie_chart=df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
				st.write(pie_chart)
				st.pyplot()

	elif option == 'Model':
		st.subheader("Model Building")
		data=st.file_uploader("Upload dataset to build model",type=['csv','xlsx','txt','json'])

		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			st.success("Now you can build your ML model!")

			if st.checkbox("Select Multiple columns"):
				new_data=st.multiselect("select columns for model building. NB:Ensure your target column is selected last",df.columns)
				df1=df[new_data]
				st.dataframe(df1)

				#splitting independent and dependent variables(X and y)
				X=df1.iloc[:,0:-1]
				y=df.iloc[:,-1]

			seed=st.sidebar.slider("Seed",1,200)

			#build the model
			classifier_name = st.sidebar.selectbox("Select your preferred Classifier",('KNN','SVM','LogisticRegression','Naive Bayes','Decision Tree'))

			def add_parameter(name_of_clf):
				params=dict()
				if name_of_clf == 'KNN':
					K=st.sidebar.slider('K',1,15)
					params['K']=K
					return params
				if name_of_clf == 'SVM':
					C = st.sidebar.slider('C',0.01,15.0)
					params['C']=C
					return params
			params=add_parameter(classifier_name)

			#get the parameter by a function
			def get_classifier(name_of_clf,params):
				clf=None
				if name_of_clf == 'KNN':
					clf=KNeighborsClassifier(n_neighbors=params["K"])
				elif name_of_clf == 'SVM':
					clf=SVC(C=params['C'])
				elif name_of_clf == 'LogisticRegression':
					clf=LogisticRegression()
				elif name_of_clf == 'Naive Bayes':
					clf=GaussianNB()
				elif name_of_clf == 'Decision Tree':
					clf=DecisionTreeClassifier()
				else:
					st.warning('Select your choice of algorithm')
				return clf
			clf = get_classifier(classifier_name,params)

			#split the data into train test split
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=seed)

			clf.fit(X_train,y_train)
			y_preds=clf.predict(X_test)
			st.write("Predictions:",y_preds)

			accuracy=accuracy_score(y_test,y_preds)
			st.write('Name of classifier:',classifier_name)
			st.write('Accuracy of Model:',accuracy)

	elif option == "Background":
		st.write("This is an Interactive web app for Data Analysis and Machine Learning building designed by Emmanuel Azi-love, feel free to reach out: aziloveemma@gmail.com")

		st.balloons()

if __name__ == '__main__':
	main()










