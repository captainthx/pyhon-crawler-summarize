import asyncio
import os
import sys
import tempfile
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from crawl4ai import *
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import streamlit as st
from langchain_community.document_loaders import UnstructuredMarkdownLoader

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

ddgs = DDGS()

OLLAMA_HOST = "http://localhost:11434"


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]

    if not with_context:
        messages.pop(0)
        messages[0]["content"] = prompt

    response = ollama.Client(host=OLLAMA_HOST).chat(
        model="gpt-oss:20b", stream=True, messages=messages
    )

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST, model_name="bge-m3:567m"
    )

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,
    )


def normalize_url(url: str):
    normalize_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "")
        .replace("-", "")
        .replace(".", "")
    )
    print(f"nomalize URL: {normalize_url}")
    return normalize_url


def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()

    for result in results:
        documents, metadata, ids = [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", ".", "?", "!", " ", ""],
        )

        if result._markdown:
            markdown_result = result._markdown.fit_markdown
        else:
            continue

        # ใช้ context manager ปิดไฟล์อัตโนมัติ
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", encoding="utf-8", delete=False
        ) as temp_file:
            temp_file.write(markdown_result)
            temp_file.flush()
            temp_path = temp_file.name

        # โหลด markdown หลังจากไฟล์ปิดเรียบร้อยแล้ว
        loader = UnstructuredMarkdownLoader(temp_path, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)

        # ลบไฟล์ออกหลังใช้เสร็จ
        os.unlink(temp_path)

        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadata.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection))
            collection.add(documents=documents, metadatas=metadata, ids=ids)


async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:

    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000,  # in ms: 20 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config)
        return results


def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []

    for url in urls:
        try:
            robot_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robot_url)
            rp.read()

            if rp.can_fetch("*", url):
                allowed_urls.append(url)
        except Exception:
            # if robots.txt is missing or there's any error, assume URL is allowed
            allowed_urls.append(url)

    return allowed_urls


def search_internet(search: str, num_result: int = 10) -> list[str]:
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            search += f" -site:{url}"

        results = ddgs.text(
            query=search,
            max_result=num_result,
            backend="bing, brave, google, yahoo, wikipedia",
        )
        results = [result["href"] for result in results]
        return check_robots_txt(results)
    except Exception as e:
        error_msg = ("❌ Failled to fetch resylts from the web", str(e))
        print(error_msg)
        st.write(error_msg)
        st.stop()


async def display():
    st.header("LLM Search crawler ⚡")
    prompt = st.text_input("search", placeholder="input search keyword")
    is_web_search = st.toggle("enable web search", value=False, key="enable_web_search")
    btn = st.button("search", icon="🔍")

    collection, chroma_client = get_vector_collection()

    if prompt and btn:
        with st.spinner(text="wait for result...", show_time=True):
            if is_web_search:
                web_result = search_internet(prompt)
                if not web_result:
                    st.write("No resutls found.")
                    st.stop()
                results = await crawl_webpages(urls=web_result, prompt=prompt)
                add_to_vector_database(results)

                qresult = collection.query(query_texts=[prompt], n_results=10)
                context = qresult.get("documents")[0]

                chroma_client.delete_collection(
                    name="web_llm"
                )  # Delete collection after use

                llm_response = call_llm(
                    context=context, prompt=prompt, with_context=is_web_search
                )
                st.write_stream(llm_response)
            else:
                llm_response = call_llm(prompt=prompt, with_context=is_web_search)
                st.write_stream(llm_response)
        st.stop()


if __name__ == "__main__":
    asyncio.run(display())
