import asyncio
import re
from crawl4ai import *
import json
from ddgs import DDGS
from ollama import Client, ChatResponse
import streamlit as st

# config Crawl4AI
browserConfig = BrowserConfig(verbose=True)
runConfig = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            # Adjust for stricter or looser results
            bm25_threshold=1.2
        )
    ),
    excluded_tags=[
        "form",
        "header",
        "footer",
        "nav",
        "aside",
        "sup",
        "ol",
        "ul",
    ],
    word_count_threshold=50,
    cache_mode=CacheMode.BYPASS,
    exclude_external_links=True,
    remove_overlay_elements=True,
    verbose=True,
)

ddgs = DDGS()

llmClient = Client(
    host="http://localhost:11434",
)


def searchInternet(query):
    results = ddgs.text(
        query=query, max_result=4, backend="bing, brave, google, yahoo, wikipedia"
    )
    filterResult = [result for result in results if "body" in result]
    return summarizeInternetSearch(query, filterResult)


def cleanUrl(text):
    urls = re.findall(r"https?://\S+", text)
    return urls[0] if urls else None


def summarizeInternetSearch(query, results):
    prompt = f"""
        You are a helpful assistant specialized in selecting the single most relevant search result to answer a user's query.
        Query: "{query}"
        Here are the search results:
        {json.dumps(results, indent=2)}
        Your task:
        1. Analyze the query and the provided search results.
        2. Select exactly ONE result that best matches the query.
        3. Output ONLY the value of the 'href' field from that result.
        4. Do NOT include any explanation, reasoning, JSON objects, or extra text. 
        5. Your final answer must be a single URL string only.
    """
    response: ChatResponse = llmClient.chat(
        model="llama3:8b", messages=[{"role": "system", "content": prompt}]
    )
    url = response.message.content.strip()
    print(f"llm select clean url: {cleanUrl(url)}")
    return cleanUrl(url)


def summarizeContent(content, query):
    prompt = f"""
    Summarize the following {content} into a concise paragraph that directly addresses the {query}. Ensure the summary 
    highlights the key points relevant to the query while maintaining clarity and completeness.
    """
    response = llmClient.chat(
        model="mixtral:8x7b",
        messages=[{"role": "system", "content": prompt}],
    )
    return response.message.content.strip()


async def crawler(url):
    async with AsyncWebCrawler() as crawler:
        result: CrawlResult = await crawler.arun(url=url, config=runConfig)
        if result.success:
            # Print clean content
            print(f"Raw : {len(result._markdown.raw_markdown)} chars")
            print(f"Fit: {len(result._markdown.fit_markdown)} chars")
            print(f"Found {len(result.media)} media items.")

            # images = result.media.get("image", [])
            # imagesResult = []
            # print("Images found:", len(images))
            # for i, img in enumerate(images):
            #     if img.get("score") == 5:
            #         imagesResult.append({"src": img["src"], "alt": img.get("alt", "")})
            #         # print(
            #         #     f"  - {img['src']} (alt={img.get('alt', '')}, score={img.get('score', 'N/A')}, group_id={img.get('group_id', '')}"
            #         # )
            # print(f"images  len:{len(imagesResult)}")
            # print(f"result crawler : {result._markdown.raw_markdown.strip()}")
            return (
                result._markdown.raw_markdown.strip(),
                result._markdown.fit_markdown.strip(),
            )
        else:
            print(f"Crawl failed: {result.error_message}")


async def main(keyword: str):
    restultWeb = searchInternet(keyword)
    rawContent, fitContent = await crawler(restultWeb)
    response = summarizeContent(rawContent, keyword)
    if response == "NOT_RELEVANT":
        print("⚠️ Content not relevant, skipping...")
    else:
        print(response)
    fileName = f"{keyword}.md"
    with open(fileName, "w", encoding="utf8") as f:
        f.write(fitContent)
    print(f"\n\n Markdown file '{fileName}' created successfully.")


if __name__ == "__main__":
    while True:
        keyword = input("Input keyword: ")
        if keyword.lower() in ["ex", "q"]:
            break
        asyncio.run(main(keyword))
