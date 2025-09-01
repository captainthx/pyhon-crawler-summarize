from ddgs import DDGS
import os
import tempfile
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from crawl4ai import *
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from dotenv import load_dotenv

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
EMBEDING_MODEL = os.getenv("EMBEDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

ddgs = DDGS()


def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST, model_name=EMBEDING_MODEL
    )

    chroma_client = chromadb.HttpClient()
    return (
        chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
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


def add_two_numbers(a: int, b: int) -> int:
    print(f"add two number tool: {a , b}")
    """
    Tool Name: add two numbers
    Description: Add two integers together and return the result.
    Args:
        a(int): frist nubmer
        b(int): second number
    Returns:
        int: sum of the two numbers
    """
    return a + b


async def search(prompt: str) -> str:
    print(f"search tool : {prompt}")
    """
    Tool Name: search
    Description: Search the internet for the given query, crawl relevant webpages,
    and return extracted raw context text. Use this tool when the user asks about
    facts, knowledge, or events that are not already in context.

    Args:
        prompt (str): search term from the user query

    Returns:
        str: raw context text extracted from web sources
    """

    collection, chroma_client = get_vector_collection()

    web_result = search_internet(prompt)
    results = await crawl_webpages(urls=web_result, prompt=prompt)
    add_to_vector_database(results)

    qresult = collection.query(query_texts=[prompt], n_results=10)
    context = qresult.get("documents")[0]

    chroma_client.delete_collection(name="web_llm")  # Delete collection after use

    return context


available_functions = {"add_two_numbers": add_two_numbers, "search": search}
