import pickle


def import_data(filename):
    """
    Import data from a file
    """
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    return data


def export_data(data, filename):
    """
    Export data to a file
    """
    file = open(filename, "wb")
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
