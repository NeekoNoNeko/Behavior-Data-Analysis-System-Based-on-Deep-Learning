from openai import OpenAI
from prompt import react_system_prompt_template
from out_1 import cut_word
from get import pos_get




class SimpleAPIAgent:
    def __init__(self, model: str = "deepseek/deepseek-v3.2"):
        self.model = model
        # 使用你的API密钥直接初始化客户端
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='输入密钥'
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





def main():
    # 创建AI助手实例
    assistant = SimpleAPIAgent()

    print("🤖 AI助手已启动! 输入'退出'结束对话")

    while True:

        # 获取用户输入
        user_input = input("\n请输入0进入坐姿时间戳获取,输入1进入ai姿态分析")
        #姿态获取
        if user_input == '0':
            pos_get('test.mp4')
            print("已获得坐姿时间戳")

        #姿态分析
        elif user_input == '1':
            txt_input = open('D:/programm/Jupyter/DL/output.txt', 'r', encoding='utf-8').read()
            print("读取到的文本内容为:", txt_input)
            try:
                reply = assistant.chat(txt_input)
                print(f"AI: {reply}")
                #写入txt
                adree = 'D:/programm/Jupyter/DL/output1.txt'
                a1 = cut_word(adree, reply)
                a1.write()
                print("已获得姿态分析")
            except Exception as e:
                print(f"出错了: {e}")


        # 检查是否退出
        else:
            print("再见! 👋")
            break

# 运行程序
if __name__ == "__main__":
    main()







