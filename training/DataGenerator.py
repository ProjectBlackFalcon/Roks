
def get_disk_posts(irrelevant=False):
    with open("../data/{}posts.txt".format("irrelevant_" if irrelevant else ""), encoding="utf-8") as reader:
        data = reader.read().split("\n")
    return data

