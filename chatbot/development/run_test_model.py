DATAFILE = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\data\angry.json"
modelpath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_model.h5"
wordspath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_words.pkl"
classpath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_classes.pkl"


from test_model import ModelTester

testobj = ModelTester(DATAFILE, modelpath, wordspath, classpath)
result = testobj.chat()

