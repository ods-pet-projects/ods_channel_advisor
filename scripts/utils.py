import pickle


def saveit(obj, ofile_path):
    with open(ofile_path, 'wb') as ofile:
        pickle.dump(obj, ofile)


def loadit(path):
    with open(path, 'rb') as ifile:
        obj = pickle.load(ifile)
        return obj