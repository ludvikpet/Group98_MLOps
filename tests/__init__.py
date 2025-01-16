import os 

_TEST_ROOT = os.path.dirname(__file__) #will get the root of tests, ie. mlops_individual_work/tests
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT) #will find the root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT,"data") #locates data path 
_VOCAB_SIZE = 30522 #should in principle also be stored in model config