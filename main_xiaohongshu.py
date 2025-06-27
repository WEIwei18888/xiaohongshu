import streamlit as st
from streamlit import spinner
from utils_xiaohongshu import generate_copywriting

# 设置页面配置
st.set_page_config(
    page_title="小红书爆款文案生成器",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS以确保输入框有边框
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border: 1px solid #CCCCCC;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    opneai_api_key = st.text_input("请输入你的OpenAI API密钥：", type="password")
    st.markdown("[获取Openai API密钥](https://platform.openai.com/api-keys)")

#应用标题
st.title("小红书爆款文案生成器 ✨")
st.write("生成包含5个标题和完整正文的结构化小红书内容")

st.divider()

# 选择内容类型
content_type = st.radio(
    "选择内容类型",
    ["产品推广", "经验教程", "旅行攻略", "生活记录"],
    index=0,
    horizontal=True
)

# 定义各类型的输入字段配置
form_config = {
    "产品推广": {
        "fields": ["产品名称", "产品类别", "卖点", "目标人群", "痛点", "风格要求"],
        "placeholders": {
            "产品名称": "例如：润百颜玻尿酸次抛精华",
            "产品类别": "美妆/护肤/美食/穿搭等",
            "卖点": "保湿、修复、独立包装等",
            "目标人群": "油皮/学生党/宝妈",
            "痛点": "卡粉、出油、敏感等",
            "风格要求": "种草风/测评风/干货风"
        }
    },
    "经验教程": {
        "fields": ["教程主题", "目标人群", "工具/材料", "步骤要点", "常见问题", "风格要求"],
        "placeholders": {
            "教程主题": "例如：3步画好截断式眼妆",
            "目标人群": "新手/进阶/熟练",
            "工具/材料": "粉底棒、卧蚕笔等",
            "步骤要点": "1. 粉底棒快速上妆技巧\n2. 3笔搞定卧蚕",
            "常见问题": "容易晕妆怎么办？卧蚕画不好？",
            "风格要求": "简洁明了/幽默风趣"
        }
    },
    "旅行攻略": {
        "fields": ["目的地", "旅行天数", "必去景点", "美食推荐", "实用Tips", "风格要求"],
        "placeholders": {
            "目的地": "例如：青岛老城区、新疆伊犁",
            "旅行天数": "2天1夜、5天4晚",
            "必去景点": "信号山、大学路等",
            "美食推荐": "本地人私藏咖啡馆、老字号餐厅",
            "实用Tips": "交通方式、门票预约、避雷提醒",
            "风格要求": "攻略型/体验型/摄影指南"
        }
    },
    "生活记录": {
        "fields": ["主题关键词", "场景细节", "时间/地点", "情感基调", "想传达的点", "风格要求"],
        "placeholders": {
            "主题关键词": "例如：独居生活、情侣日常",
            "场景细节": "周六上午去胡同咖啡店看书的细节",
            "时间/地点": "周末、北京胡同",
            "情感基调": "治愈、搞笑、感动",
            "想传达的点": "享受独处、记录生活",
            "风格要求": "文艺清新/真实记录"
        }
    }
}

# 动态生成输入表单
with st.form("input_form"):
    st.subheader(f"✍️ 填写{content_type}信息")
    col1, col2 = st.columns(2)

    user_input = {}
    current_fields = form_config[content_type]["fields"]
    placeholders = form_config[content_type]["placeholders"]

    #借助 enumerate() 函数能够同时获取索引 i 和元素 field。
    for i, field in enumerate(current_fields):
        if i % 2 == 0:   #i % 2: 这里的 % 是取模运算符，它返回 i 除以 2 得到的余数
            with col1:
                user_input[field] = st.text_area(
                    field,
                    placeholder=placeholders[field],
                    key=f"{content_type}_{field}",#它确保了在不同内容类型下，同名字段也能有唯一的标识符。当用户切换内容类型（如从 "产品推广" 到 "旅行攻略"）时，Streamlit 无法区分不同类型下的同名字段（如 "产品名称" 和 "目的地"），导致输入内容混乱，甚至可能引发异常
                    height=100
                )
        else:
            with col2:
                user_input[field] = st.text_area(
                    field,
                    placeholder=placeholders[field],
                    key=f"{content_type}_{field}",
                    height=100
                )

    generate_button = st.form_submit_button("生成文案", use_container_width=True)


#生成文案逻辑
if generate_button and not opneai_api_key:
    st.info("请输入你的Open AI密钥！！！")
    st.stop()
if generate_button:
    # 定义每种内容类型的必填字段
    required_fields = {
        "产品推广": ["产品名称", "产品类别", "卖点", "目标人群", "痛点", "风格要求"],
        "经验教程": ["教程主题", "目标人群", "工具/材料", "步骤要点", "常见问题", "风格要求"],
        "旅行攻略": ["目的地", "旅行天数", "必去景点", "美食推荐", "实用Tips", "风格要求"],
        "生活记录": ["主题关键词", "场景细节", "时间/地点", "情感基调", "想传达的点", "风格要求"]
    }

    # 检查所有必填字段
    missing_fields = []
    for field in required_fields[content_type]:
        if not user_input.get(field):
            missing_fields.append(field)

    # 显示错误信息
    if missing_fields:
        field_text = "、".join(missing_fields)#把列表转换成字符串
        st.info(f"请填写以下必填信息：{field_text}！")
    else:
        with spinner("AI正在努力创作中，请稍等..."):
            response = generate_copywriting(opneai_api_key, content_type, user_input)

        #显示标题
        st.markdown("#### 🌟 生成的5个标题")
        for i, title in enumerate(response.title):
            st.markdown(f"**标题{i+1}**：{title}")

        # 显示正文
        st.markdown("#### 📝 完整正文内容")
        st.markdown(response.content)