This folder contains the following:
Dockerfile : The main docker file     
requirements.txt: Docker file depenedencies
app folder:
KNN_model.pkl: Pickled KNN model  
Random_Forest_model.pkl : Pickled Random Forest model 
main.py : The Main fastAPI program
LR_model.pkl : Pickled  Logistic Regression model
SVM_model.pkl :Pickled Support Vector Machine model           
testScript.txt: test script containing curl commands for testing

Recommended to use Git Bash. (Should also work from windows CMD shell)
Build the docker image :
docker build . -t uvicorn
Run the docker image:
docker run -it -p 80:80 uvicorn
