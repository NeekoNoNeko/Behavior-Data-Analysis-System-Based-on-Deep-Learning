
class cut_word:
    def __init__(self, adree, word):
        self.adree = adree
        self.word = word


    def write(self):
        with open(self.adree, 'w', encoding='utf-8') as file:
            file.write(self.word)
        print(f"文件已写入：{self.adree}")



