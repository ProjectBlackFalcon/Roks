
def get_disk_posts():
    with open("../data/posts.txt", encoding="utf-8") as reader:
        data = reader.read().split("\n")
    return data

