import sys
from PreProcess import PreProcessor
from Train import Trainer

def train():
    train = Trainer()
    # train.loadData()
    train.execute()
    

def test():
    pass

def trial():
    train = Trainer()
    train.trial()

def preprocess():
    
    preProcess = PreProcessor()
   
    preProcess.execute()
    preProcess.save()

if __name__ == "__main__":
    args = sys.argv
    
    if "train" in args:
        train()
        sys.exit()
    elif "test" in args:
        test()
        sys.exit()
    elif "preprocess" in args:
        preprocess()
        sys.exit()
    elif "trial" in args:
        trial()
        sys.exit()
    else:
        print("No valid args!!")
        sys.exit()