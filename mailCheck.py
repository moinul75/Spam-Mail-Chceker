import pandas as pd 
from  sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('emails.csv')
#Data Cleaning
df.drop_duplicates(inplace=True)
df.isnull().sum()
x = df.text.values
y = df.spam.values

#Data Preprocessing 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2) #get 20% data for test 
#get text data to numeric 
cv = CountVectorizer()
x_train = cv.fit_transform(xtrain)

model = MultinomialNB()
model.fit(x_train,ytrain)
#Test the model 
x_test = cv.transform(xtest)
model.score(x_test,ytest)


#operation for working any input values 
while True:
    print("="*10+"Spam Mail Detection"+"="*10)
    email = [input("Enter your mail: ")]
    cv_email = cv.transform(email)
    ans = model.predict(cv_email)
    if ans == [0]:
        print("Valid Email..Thanks")
    elif ans == [1]:
        print("Opps Its a spam mail..")
    else:
        break
    

 


