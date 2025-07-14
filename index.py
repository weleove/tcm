import logging
import sys
import torch
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr


# 读取本地 .env 文件
_ = load_dotenv(find_dotenv(), verbose=True)  
client = OpenAI()

# 设置日志记录
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 定义System Prompt
SYSTEM_PROMPT = """你是一个医疗人工智能助手。对于用户的提问，如果与医学无关，或者检索到的内容相似度较低，请直接回答：抱歉，我对于您的请求“{query_str}”不了解"""
query_wrapper_prompt = PromptTemplate(
 "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

# # 使用 llama_index_llms_huggingface 调用本地大模型
# llm = HuggingFaceLLM(
#  context_window=4096,
#  max_new_tokens=2048,
#  generate_kwargs={"temperature": 0.0, "do_sample": False},
#  query_wrapper_prompt=query_wrapper_prompt,
#  tokenizer_name='modelscope\Qwen\Qwen2___5-7B-Instruct',
#  model_name='modelscope\Qwen\Qwen2___5-7B-Instruct',
#  device_map="auto",
#  model_kwargs={"torch_dtype": torch.float16},
# )
# Settings.llm = llm

# 调用远程大模型
llm = LlamaIndexOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=2048,
    system_prompt=SYSTEM_PROMPT,
    api_base="https://api.openai-proxy.org/v1",
)
Settings.llm = llm

# 调用本地 embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
 model_name="modelscope\\BAAI\\bge-base-zh-v1___5"
)

# 读取文档
documents = SimpleDirectoryReader("./books", required_exts=[".txt"]).load_data()


# # 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
# index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])
# # SentenceSplitter 参数详细设置：
# # 预设会以 1024 个 token 为界切割片段, 每个片段的开头重叠上一个片段的 200 个 token 的内容。
# # chunk_size=1024, # 切片 token 数限制
# # chunk_overlap=200, # 切片开头与前一片段尾端的重复 token 数
# # paragraph_separator='\n\n\n', # 段落的分界
# # secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?' # 单一句子的样式
# # separator=' ', # 最小切割的分界字元

# 采用chroma作为向量存储
# 创建 Chroma 客户端
chroma_client = chromadb.PersistentClient("./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("tcm_knowledge_base")


# 初始化 ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建 StorageContext 并传入 vector_store
storage_context = StorageContext.from_defaults(persist_dir="./chroma_db/tcm_knowledge_base", vector_store=vector_store)

# 构建索引
import os


index_dir = "./chroma_db/tcm_knowledge_base"

if not os.path.exists(index_dir):
    # 第一次：构建并保存索引
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[SentenceSplitter(chunk_size=256)],
        storage_context=storage_context,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=index_dir)
else:
    # 已有索引，直接加载
    index = load_index_from_storage(
        storage_context,
    )

 
# 请求内容
# query = "请问《黄帝内经》中的阴阳五行理论是如何影响中医诊断和治疗的？"

# 构建查询引擎
# query_engine = index.as_query_engine(similarity_top_k=5)

# 获取相似度 top 5 的片段
# contexts = query_engine.retrieve(QueryBundle(query))
# print('-'*10 + 'ref' + '-'*10)
# for i, context in enumerate(contexts):
#  print('*'*10 + f'chunk {i} start' + '*'*10)
#  content = context.node.get_content(metadata_mode=MetadataMode.LLM)
#  print(content)
#  print('*' * 10 + f'chunk {i} end' + '*' * 10)
# print('-'*10 + 'ref' + '-'*10)

# 多轮对话记忆
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# 构建带记忆的查询引擎
query_engine = index.as_chat_engine(
    chat_mode="context",
    llm=llm,
    memory=memory,
    system_prompt=SYSTEM_PROMPT,
    similarity_top_k=5
)

# 用于 Gradio 的聊天历史
chat_history = []

def chat_interface(user_input):
    global chat_history
    if not user_input.strip():
        return chat_history

    # 获取模型回复
    response = query_engine.chat(user_input)

    # 记录历史
    chat_history.append(["用户：" + user_input, "AI：" + str(response)])
    return chat_history, ''

# 创建 Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("# 中医问答助手 ")
    chatbot = gr.Chatbot(label="对话窗口")
    msg = gr.Textbox(label="请输入您的问题", placeholder="例如：请解释阴阳五行如何影响诊断？")
    with gr.Row():
        submit_btn = gr.Button("提交", variant="primary")
        clear_btn = gr.Button("清空")

    def clear_chat():
        global chat_history
        chat_history = []
        return [], ''

    submit_btn.click(fn=chat_interface, inputs=msg, outputs=[chatbot, msg])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg])
    

# 启动服务
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)  # share=True 可公网访问




# # 进行查询
# response = query_engine.query(query)
# print(response)


