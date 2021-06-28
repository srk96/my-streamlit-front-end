import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import altair as alt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image 

from sklearn.model_selection import train_test_split




header=st.beta_container()
dataset=st.beta_container()
interactive=st.beta_container()
model_training=st.beta_container()


st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
  )

background_color= '#F5F5F5'

@st.cache
def get_data(filename):
	mental_health_data= pd.read_csv(filename)

	return mental_health_data




with header:
	st.title('Welcome to my Healthcare Analytics Project !')
	st.title('Mental Health in Tech Workplace')
	st.text('In this project, certain attitudes and behaviors towards mental health in the Tech workplace are explored.')


with dataset:

	mental_health_data = get_data('data/Ment_Health_in_Tech_Workplace.csv')

	picture_col,data_col=st.beta_columns(2)

	image=Image.open('Images/Mental_Health_Awareness.jpg')
	picture_col.image(image, use_column_width=True)

	if data_col.checkbox ('Show Mental Health in Tech Workplace dataset'):
		data_col.write(mental_health_data, use_column_width=True)

	st.text('This survey dataset is sourced from Kaggle.com. The original dataset could be found on Open Sourcing Mental Illness site.')

	
	st.header('A General Look into the Dataset')



	st.subheader('Any family history of mental illness ?')
	
	family_history_dist=pd.DataFrame(mental_health_data['family_history'].value_counts())

	family_history_dist=family_history_dist.reset_index()
	family_history_dist.columns=['family_history','count']

	


	fig=px.bar(family_history_dist,x='family_history', y='count', color_discrete_sequence=['#7B241C'])

	fig.update_layout(showlegend=True,paper_bgcolor= background_color, font=dict(color='#383635', size= 15))
     

	st.plotly_chart(fig)


	




	



	st.subheader('Is mental health interfering with work within a Tech workplace ?')

	
	fig=go.Figure(data=go.Table(
	   header=dict(values=list(mental_health_data[['Age','tech_company','no_employees','work_interfere']].columns),
	      fill_color='#E6B0AA', 
	      align='center'), 
	   cells=dict(values=[mental_health_data.Age.sort_values(),mental_health_data.tech_company, mental_health_data.no_employees, mental_health_data.work_interfere],
		  fill_color='#E5ECF6',
		  align='left')))	

	fig.update_layout(margin=dict(l=5, r=5,b=10, t=10),paper_bgcolor= background_color)

	st.write(fig)


	





	st.header('A closer look into the data')

	


	

	pie_chart_1_col,pie_chart_2_col,pie_chart_3_col=st.beta_columns(3)

	pie_chart_1_col.subheader('Discussing a mental health issue with the employer would have negative consequences ?')

	mental_health_conseq_dist=pd.DataFrame(mental_health_data['mental_health_consequence'].value_counts())


	mental_health_conseq_dist=mental_health_conseq_dist.reset_index()
	mental_health_conseq_dist.columns=['mental_health_consequence','count']

	fig=px.pie(mental_health_conseq_dist, values='count', names='mental_health_consequence', hover_name='mental_health_consequence',color='mental_health_consequence', color_discrete_map={'Maybe':'#641E16',
                                 'No':'#A93226',
                                 'Yes':'#D98880'})

	fig.update_layout(showlegend=True,width=480, height=480, paper_bgcolor= background_color, font=dict(color='#383635', size= 15), 
		margin=dict(l=1, r=1,b=1,t=1))

	pie_chart_1_col.plotly_chart(fig)

	pie_chart_3_col.subheader('Discussing a physical health issue with the employer would have negative consequences ?')


	phys_health_conseq_dist=pd.DataFrame(mental_health_data['phys_health_consequence'].value_counts())

	phys_health_conseq_dist=phys_health_conseq_dist.reset_index()
	phys_health_conseq_dist.columns=['phys_health_consequence','count']

	fig=px.pie(phys_health_conseq_dist, values='count', names='phys_health_consequence', hover_name='phys_health_consequence', color='phys_health_consequence', color_discrete_map={'Maybe':'#641E16',
                                 'No':'#A93226',
                                 'Yes':'#D98880'})

	fig.update_layout(showlegend=True,width=480, height=480, paper_bgcolor= background_color, font=dict(color='#383635', size= 15), 
		margin=dict(l=1, r=1,b=1,t=1))

	pie_chart_3_col.plotly_chart(fig)





	bar_chart_1,bar_chart_2, bar_chart_3=st.beta_columns(3)


	bar_chart_1.subheader('Has the employer ever discussed mental health as part of an employee wellness program ?')
	
	wellness_program_dist=pd.DataFrame(mental_health_data['wellness_program'].value_counts())

	wellness_program_dist=wellness_program_dist.reset_index()

	wellness_program_dist.columns=['wellness_program','count']

	fig=px.bar(wellness_program_dist,x='wellness_program', y='count', color_discrete_sequence=['#F0B27A'])

	fig.update_layout(showlegend=True, paper_bgcolor= background_color, font=dict(color='#383635', size= 15))

	bar_chart_1.plotly_chart(fig)



	bar_chart_3.subheader('Does the employer provide resources to learn more about mental health issues and how to seek help ?')
	
	seek_help_dist=pd.DataFrame(mental_health_data['seek_help'].value_counts())

	seek_help_dist=seek_help_dist.reset_index()

	seek_help_dist.columns=['seek_help','count']

	fig=px.bar(seek_help_dist,x='seek_help', y='count', color_discrete_sequence=['#F0B27A'])

	fig.update_layout(paper_bgcolor= background_color, font=dict(color='#383635', size= 15))

	bar_chart_3.plotly_chart(fig)


	pie_chart_1_col,pie_chart_2_col, pie_chart_3_col=st.beta_columns(3)

	pie_chart_1_col.subheader('Is a mental health issue discussed with coworkers ?')

	coworkers_dist=pd.DataFrame(mental_health_data['coworkers'].value_counts())


	coworkers_dist=coworkers_dist.reset_index()
	coworkers_dist.columns=['coworkers','count']

	fig=px.pie(coworkers_dist, values='count', names='coworkers', hover_name='coworkers', color='coworkers', color_discrete_map={'No':'#1F618D',
                                 'Some of them':'#154360',
                                 'Yes':'#AED6F1'})

	fig.update_layout(showlegend=True,width=520, height=520, paper_bgcolor= background_color,font=dict(color='#383635', size= 15), 
		margin=dict(l=1, r=1,b=1,t=1))

	pie_chart_1_col.plotly_chart(fig)




	pie_chart_3_col.subheader('Is there any observed negative consequences for coworkers with mental health conditions in the workplace ?')


	obs_conseq_dist=pd.DataFrame(mental_health_data['obs_consequence'].value_counts())

	obs_conseq_dist=obs_conseq_dist.reset_index()
	obs_conseq_dist.columns=['obs_consequence','count']

	fig=px.pie(obs_conseq_dist, values='count', names='obs_consequence', hover_name='obs_consequence', color='obs_consequence',color_discrete_map={'No':'#154360',
                                 'Yes':'#1F618D'})

	fig.update_layout(showlegend=True,width=450, height=450, paper_bgcolor= background_color, font=dict(color='#383635', size= 15), 
		margin=dict(l=1, r=1,b=1,t=1))

	pie_chart_3_col.plotly_chart(fig)







	st.subheader('Is employees anonymity protected if they choose to take advantage of mental health treatment ?')

	df= pd.DataFrame(mental_health_data, columns=['Age','Gender','anonymity'])


	x = alt.Chart(df).mark_circle().encode(x='Age', y='Gender', size='anonymity', color='anonymity', tooltip=['Age', 'Gender', 'anonymity'])

	st.altair_chart(x, use_container_width=True)




	

	st.subheader('Does the employer take mental health as seriously as physical health ?')


	df= pd.DataFrame(mental_health_data, columns=['Age','Country','mental_vs_physical'])


	c = alt.Chart(df).mark_circle().encode(x='Age', y='Country', size='mental_vs_physical', color='mental_vs_physical', tooltip=['Age', 'Country', 'mental_vs_physical'])

	st.altair_chart(c, use_container_width=True)









	st.subheader('How easy is it to take medical leave for a mental health condition ?')


	line_chart_data=mental_health_data.copy()

	age_cross_tab=pd.crosstab(line_chart_data['Age'], line_chart_data['leave'])

	print(age_cross_tab)

	fig=px.line(age_cross_tab)

	

	fig.update_layout(showlegend=True,width=900,margin=dict(l=10,r=10,b=10,t=10),paper_bgcolor=background_color,font=dict(color='#383635', size=15))

	st.plotly_chart(fig)



	



train, test = train_test_split(mental_health_data, test_size=0.2, random_state=1111)

train.reset_index(inplace=True)

with model_training:
	st.title ('Time to train the model !')
	st.text('Here you get to choose the hyperparameters of different models and notice the performance variability.')

	mental_health_data_encoded=pd.get_dummies(train,columns=['Gender', 'Country', 'family_history',  'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers',  'obs_consequence'])


	if st.checkbox ('Show Mental Health in Tech Workplace Encoded dataset'):
		st.subheader('Mental Health in Tech Workplace')
		st.write(mental_health_data_encoded)

	st.text('The dataset was encoded for predictive analytics purposes.')


	st.header('Random Forest Classifier')                     

	
	sel_col, disp_col=st.beta_columns(2)

	max_depth=sel_col.slider('What should be the max_depth of the model?', min_value=2, max_value=8, step=2)

	n_estimators=sel_col.selectbox ('How many trees should there be?', options=[100,200,300], index=0)

	criterion = sel_col.selectbox('What is the quality of the split?', options=['gini', 'entropy'], index=0)

	max_features=sel_col.selectbox('What is the number of features to consider when looking for the best split?', options=['auto', 'sqrt', 'log2'], index=0)

	sel_col.text('Here is a list of features in my dataset:')

	sel_col.write(mental_health_data.columns)

	input_feature=sel_col.text_input('Which feature should be used as the input feature?', 'mental_health_consequence', key='input_feature')


	forest_classifier=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, criterion=criterion, max_features=max_features)

	

	X=mental_health_data_encoded.drop('mental_vs_physical', axis=1)

	y=mental_health_data_encoded[['mental_vs_physical']]

	forest_classifier.fit(X,y)

	prediction=forest_classifier.predict(X)

	disp_col.subheader('Accuracy score of the model is:')

	disp_col.write(accuracy_score(y,prediction))




	st.header('Decision Tree Classifier')

	sel_col, disp_col=st.beta_columns(2)


	max_depth=sel_col.slider('What should be the max_depth of the model?', min_value=2, max_value=8, step=2, key='max_depth')

	criterion = sel_col.selectbox('What is the quality of the split?', options=['gini', 'entropy'], index=0, key='criterion')

	max_features=sel_col.selectbox('What is the number of features to consider when looking for the best split?', options=['auto', 'sqrt', 'log2'], index=0, key='max_features')

	min_samples_split=sel_col.slider('What is the minimum number of samples required to split an internal node?', min_value=2, max_value=8, step=2, key='min_samples_split')

	splitter=sel_col.selectbox('What is the strategy used to choose the split at each node?', options=['best', 'random'], index=0, key='splitter')

	sel_col.text('Here is a list of features in my dataset:')

	sel_col.write(mental_health_data.columns)

	input_feature_1=sel_col.text_input('Which feature should be used as the input feature?', 'mental_health_consequence', key='input_feature_1')

	
	tree_classifier=DecisionTreeClassifier(max_depth=max_depth, min_samples_split=n_estimators, criterion=criterion, max_features=max_features, splitter=splitter)

	X=mental_health_data_encoded.drop('mental_vs_physical', axis=1)

	y=mental_health_data_encoded[['mental_vs_physical']]

	tree_classifier.fit(X,y)

	prediction=tree_classifier.predict(X)

	disp_col.subheader('Accuracy score of the model is:')

	disp_col.write(accuracy_score(y,prediction))




	st.header('Gradient Boosting Classifier')

	sel_col, disp_col=st.beta_columns(2)



	max_depth=sel_col.slider('What should be the max_depth of the model?', min_value=3, max_value=10, step=1, key='max_depth')

	learning_rate = sel_col.slider('How quickly is the error corrected from each tree to the next?', min_value=0.1, max_value=1.0, step=0.1, key='learning_rate')

	n_estimators=sel_col.selectbox ('How many boosting stages should there be?', options=[100,200,300,400], index=0, key='n_estimators')

	loss=sel_col.selectbox('What is the optimal loss function?', options=['deviance', 'exponential'], index=0, key='loss')

	sel_col.text('Here is a list of features in my dataset:')

	sel_col.write(mental_health_data.columns)

	input_feature_2=sel_col.text_input('Which feature should be used as the input feature?', 'mental_health_consequence', key='input_feature_2')


	gradient_classifier=GradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate, loss=loss, n_estimators=n_estimators)

	X=mental_health_data_encoded.drop('mental_vs_physical', axis=1)

	y=mental_health_data_encoded[['mental_vs_physical']]

	gradient_classifier.fit(X,y)

	prediction=tree_classifier.predict(X)

	disp_col.subheader('Accuracy score of the model is:')

	disp_col.write(accuracy_score(y,prediction))


	














	


	
    


	













	






	
    

    