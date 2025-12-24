from openai import OpenAI
from prompt import react_system_prompt_template
from get import pos_get




class SimpleAPIAgent:
    def __init__(self, model: str = "deepseek/deepseek-v3.2"):
        self.model = model
        # 使用你的API密钥直接初始化客户端
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='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'
        )

    def chat(self, user_input: str) -> str:
        """简单的聊天对话功能"""
        messages = [{"role": "system", "content": react_system_prompt_template},
            {"role": "user", "content": user_input}
        ]

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # 返回AI的回复
        return response.choices[0].message.content


    # 写入文件
    def write(self, adree, word):
        with open(adree, 'w', encoding='utf-8') as file:
            file.write(word)
        print(f"文件已写入：{adree}")



def main():
    # 创建AI助手实例
    assistant = SimpleAPIAgent()

    print("姿态检测已开启")

    while True:
        #获取坐姿时间戳
        print("获取坐姿时间戳")
        pos_get('test.mp4')
        print("已获得坐姿时间戳")

        #姿态分析
        print("姿态分析")
        txt_input = open('output.txt', 'r', encoding='utf-8').read()
        print("读取到的文本内容为:", txt_input)
        try:
            reply = assistant.chat(txt_input)
            print(f"AI: {reply}")
            #写入txt
            adree = 'output1.txt'
            assistant.write(adree, reply)
            print("已获得姿态分析")
        except Exception as e:
            print(f"出错了: {e}")

        a1 = input("是否继续检测？(y/n)")
        if a1 == 'n':
            break


# 运行程序
if __name__ == "__main__":
    main()






