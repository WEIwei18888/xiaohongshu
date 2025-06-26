from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel, Field
from typing import List



# 定义结构化输出模型
class XiaoHongShu(BaseModel):
    title: List[str] = Field(
        default_factory=list,
        description="生成5个不同风格的小红书标题，每个标题需包含emoji和关键词，长度控制在15-25字"
    )
    content: str = Field(
        default="",
        description="一篇完整的小红书正文，需包含痛点引入、亮点描述、使用体验，结尾有互动引导"
    )


# 创建解析器
output_parser = PydanticOutputParser(pydantic_object=XiaoHongShu)
parser_instructions = output_parser.get_format_instructions()

#通用System Prompt - 包含解析指令和写作指南
system_template_text = """
你是一位专业的小红书文案创作者，拥有丰富的爆款内容经验。请遵循以下准则生成内容，并严格按照指定格式输出：

【内容创作准则】
1. 标题创作：
   - 必须包含emoji符号增强吸引力（如💡✨❗️）
   - 采用「痛点+解决方案」「数字+亮点」等爆款结构
   - 每个标题需差异化，覆盖不同关键词组合

2. 正文结构：
   - 开头：用场景化描述引发共鸣（如"打工人每天早起化妆真的好累..."）
   - 中间：分点阐述核心亮点，结合具体使用场景
   - 结尾：引导互动（如"你们试过吗？评论区分享你的感受～"）

3. 标签策略：
   - 优先使用平台热门标签，包含用户指定标签
   - 格式为#关键词，避免重复或无关标签

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

    #创建链
    chain = prompt_template|model|output_parser

    # Combine user_input with parser_instructions for the invoke call
    full_input = {**user_input, "parser_instructions": parser_instructions}

    #获得回复
    response = chain.invoke(full_input)

    return response

