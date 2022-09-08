from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from treeinterpreter import treeinterpreter as ti 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

app= Flask(__name__)


df= pd.read_csv('loan_data.csv')

cols = []
for x in list(df.columns):
  cols.append(x.replace('.','_'))

df.columns = cols

'''cat_feats=['purpose']

final_data=pd.get_dummies(df,columns=cat_feats,drop_first=True)'''

# Split dataset
X= df.drop('not_fully_paid', axis=1)
y= df['not_fully_paid']

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=101)

model=RandomForestClassifier(n_estimators=300)




# First we need to know which columns are binary, nominal and numerical
def get_columns_by_category():
    categorical_mask = X.select_dtypes(
        include=['object']).apply(pd.Series.nunique) == 2
    numerical_mask = X.select_dtypes(
        include=['int64', 'float64']).apply(pd.Series.nunique) > 5

    binary_columns = X[categorical_mask.index[categorical_mask]].columns
    nominal_columns = X[categorical_mask.index[~categorical_mask]].columns
    numerical_columns = X[numerical_mask.index[numerical_mask]].columns

    return binary_columns, nominal_columns, numerical_columns

binary_columns, nominal_columns, numerical_columns = get_columns_by_category()

# Now we can create a column transformer pipeline

transformers = [('binary', OrdinalEncoder(), binary_columns),
                ('nominal', OneHotEncoder(), nominal_columns),
                ('numerical', StandardScaler(), numerical_columns)]

transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')

pipe = Pipeline([('transformer', transformer_pipeline), ('Random Forest Classifier', model)])
pipe.fit(X_train, y_train)

msg =  (
    'Whether the borrower meets the credit underwriting criteria of the company.',
    'The purpose of the loan',
    'The interest rate of the loan',
    'The monthly installments ($) owed by the borrower if the loan is funded',
    'The self-reported annual income of the borrower',
    'The debt-to-income ratio of the borrower',
    'The FICO credit score of the borrower',
    'The number of days the borrower has had a credit line.',
    'The borrower’s revolving balance (amount unpaid at the end of the credit card billing cycle)',
    'The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available).',
    'The borrower’s number of inquiries by creditors in the last 6 months',
    'The number of times the borrower had been 30+ days past due on a payment in the past 2 years',
    'The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments)')

col_msg = dict(zip(X_train.columns, msg))


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    credit_policy= request.form['credit_policy']
    purpose= request.form['purpose']
    int_rate= request.form['int_rate']
    int_rate=float(int_rate)/100
    installment= request.form['installment']
    log_annual_inc= request.form['log_annual_inc']
    log_annual_inc=np.log(int(log_annual_inc))
    dti= request.form['dti']
    fico= request.form['fico']
    days_with_cr_line= request.form['days_with_cr_line']
    revol_bal= request.form['revol_bal']
    revol_util= request.form['revol_util']
    inq_last_6mths= request.form['inq_last_6mths']
    delinq_2yrs= request.form['delinq_2yrs']
    pub_rec= request.form['pub_rec']

    arr = pd.DataFrame(np.array([[credit_policy,purpose,int_rate,installment,log_annual_inc,
        dti,fico,days_with_cr_line,revol_bal,revol_util,inq_last_6mths,delinq_2yrs,pub_rec]]), columns=X_train.columns) 
         
    pred= pipe.predict(arr)


    prediction, bias, contributions = ti.predict(pipe[-1], pipe[:-1].transform(arr))
    nums=[]
    for c, feature in sorted(zip(contributions[0], 
                                 X_test.columns), 
                             key=lambda x: ~abs(x[0]).all()):
        rnd=c[np.argmax(bias[0])]
        nums.append(rnd)

    listd = sorted(zip(X_test.columns,nums),key=lambda x: (x[1])**2)

    most = listd[-1][0]

    a,b = list(zip(*listd[-6:-1]))
    next5  = list(reversed(list(a)))


    return render_template('after.html', data=pred ,
        most=most, next5=next5, col_msg = col_msg)
        

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)