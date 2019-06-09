from pytorch_pretrained_bert import GPT2Tokenizer
from DataGenerator import get_disk_posts

enc = GPT2Tokenizer.from_pretrained("gpt2")
posts = get_disk_posts()

for index, post in enumerate(posts):
    words = post.split(" ")
    for word in words:

        try:
            enc.encode(word)
        except KeyError:
            print("Failed on", word)
            posts[index] = ''.join(posts[index].split(word))
    print(index, len(posts))

with open('clean_posts.txt', 'w+', encoding="utf-8") as file:
    file.write('\n'.join(posts))

print("Done")
