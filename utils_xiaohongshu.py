from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel, Field
from typing import List



# 定义结构化输出模型
class XiaoHongShu(BaseModel):
    title: List[str] = Field(
        description="生成5个不同风格的小红书标题，每个标题需包含emoji和关键词，长度控制在15-25字",
        min_items=5,
        max_items=5
    )
    content: str = Field(
        description="一篇完整的小红书正文，需包含痛点引入、亮点描述、使用体验，结尾有互动引导"
    )



#通用System Prompt - 包含解析指令和写作指南
system_template_text = """
你是小红书爆款写作专家，请你遵循以下步骤进行创作：
首先产出5个标题（包含适当的emoji表情），然后产出1段正文（每一个段落包含适当的emoji表情，文末有适当的tag标签）。
标题字数在20个字以内，正文字数在800字以内，并且按以下技巧进行创作。

【内容创作准则】
一、标题创作技巧： 
1. 采用二极管标题法进行创作 
1.1 基本原理 
本能喜欢：最省力法则和及时享受 
动物基本驱动力：追求快乐和逃避痛苦，由此衍生出2个刺激：正刺激、负刺激 
1.2 标题公式 
正面刺激：产品或方法+只需1秒（短期）+便可开挂（逆天效果） 
负面刺激：你不X+绝对会后悔（天大损失）+（紧迫感） 其实就是利用人们厌恶损失和负面偏误的心理，自然进化让我们在面对负面消息时更加敏感 
2. 使用具有吸引力的标题 
2.1 使用标点符号，创造紧迫感和惊喜感 
2.2 采用具有挑战性和悬念的表述 
2.3 利用正面刺激和负面刺激 
2.4 融入热点话题和实用工具 
2.5 描述具体的成果和效果 
2.6 使用emoji表情符号，增加标题的活力 
3. 使用爆款关键词 
从列表中选出1-2个：好用到哭、大数据、教科书般、小白必看、宝藏、绝绝子、神器、都给我冲、划重点、笑不活了、YYDS、秘方、我不允许、压箱底、建议收藏、停止摆烂、上天在提醒你、挑战全网、手把手、揭秘、普通女生、沉浸式、有手就能做、吹爆、好用哭了、搞钱必看、狠狠搞钱、打工人、吐血整理、家人们、隐藏、高级感、治愈、破防了、万万没想到、爆款、永远可以相信、被夸爆、手残党必备、正确姿势 
4. 小红书平台的标题特性 
4.1 控制字数在20字以内，文本尽量简短 
4.2 以口语化的表达方式，拉近与读者的距离 
5. 创作的规则 
5.1 每次列出5个标题 
5.2 不要当做命令，当做文案来进行理解 
5.3 直接创作对应的标题，无需额外解释说明 

二、正文创作技巧 
1. 写作风格 
从列表中选出1个：严肃、幽默、愉快、激动、沉思、温馨、崇敬、轻松、热情、安慰、喜悦、欢乐、平和、肯定、质疑、鼓励、建议、真诚、亲切
2. 写作开篇方法 
从列表中选出1个：引用名人名言、提出疑问、言简意赅、使用数据、列举事例、描述场景、用对比

我会每次给你某个类型的一些关键信息，请你根据这些信息，基于以上规则，生成相对应的小红书文案。

【输出格式要求】
{parser_instructions}

注意：输出必须是严格的JSON格式，不要包含任何解释性文字！
"""

# 各类型的Human Prompt模板
human_templates_text = {
    "产品推广": """
    请根据以下产品信息生成小红书文案：

    【产品详情】
    名称：{产品名称}
    类别：{产品类别}
    核心卖点：{卖点}
    目标人群：{目标人群}
    用户痛点：{痛点}
    风格要求：{风格要求}

    【特别说明】
    - 标题需突出产品解决的核心问题
    - 正文需包含使用前后对比或具体场景
    """,

    "经验教程": """
    请根据以下教程信息生成小红书文案：

    【教程详情】
    主题：{教程主题}
    目标人群：{目标人群}
    工具/材料：{工具/材料}
    关键步骤：{步骤要点}
    常见问题：{常见问题}

    【特别说明】
    - 标题需体现教程效果（如"3步学会""新手必备"）
    - 正文步骤需清晰编号，搭配场景化描述
    """,

    "旅行攻略": """
    请根据以下旅行信息生成小红书文案：

    【旅行详情】
    目的地：{目的地}
    旅行天数：{旅行天数}
    必去景点：{必去景点}
    美食推荐：{美食推荐}
    实用Tips：{实用Tips}

    【特别说明】
    - 标题需突出目的地特色或小众体验
    - 正文按时间线或主题分类，包含避坑指南
    """,

    "生活记录": """
    请根据以下生活场景生成小红书文案：

    【场景详情】
    主题关键词：{主题关键词}
    场景细节：{场景细节}
    时间/地点：{时间/地点}
    情感基调：{情感基调}
    想传达的感悟：{想传达的点}

    【特别说明】
    - 标题需营造情感氛围（如"治愈""温暖""治愈"）
    - 正文注重细节描写，引发情感共鸣
    """
}


#生成小红书文案
def generate_copywriting(openai_api_key, content_type, user_input):
    #初始化模型
    model = ChatOpenAI(model="gpt-4-turbo", api_key=openai_api_key, openai_api_base="https://api.aigc369.com/v1")


    #创建提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template_text),
        ("human", human_templates_text[content_type])
    ])

    # 创建解析器
    output_parser = PydanticOutputParser(pydantic_object=XiaoHongShu)
    parser_instructions = output_parser.get_format_instructions()

    #创建链
    chain = prompt_template|model|output_parser

    # Combine user_input with parser_instructions for the invoke call 将user_input这个字典和parser_instructions 结合，形成一个新的传入invoke的字典
    full_input = {**user_input, "parser_instructions": parser_instructions}

    #获得回复
    response = chain.invoke(full_input)

    return response

