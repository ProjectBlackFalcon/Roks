
def get_disk_posts(clean=False, irrelevant=False):
    with open("../data/{}posts.txt".format("clean_" if clean else "irrelevant_" if irrelevant else ""), encoding="utf-8") as reader:
        data = reader.read().split("\n")
    return data


def save_data(data, data_name):
    with open('../data/{}.txt'.format(data_name), 'w+', encoding="utf-8") as file:
        file.write(data)
        print("wrote", data, "to file")


def read_tokens(data_name):
    with open('../data/{}.txt'.format(data_name), 'r', encoding="utf-8") as file:
        token_string = file.read()
        tokens = token_string.split("\n")
        return [single.split("|") for single in tokens]