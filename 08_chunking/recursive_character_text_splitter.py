import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("08_chunking/docs/sistema_solar.txt", encoding="utf-8")
documents = loader.load()

code_loader = TextLoader("08_chunking/docs/quicksort.py", encoding="utf-8")
code_documents = code_loader.load()

SEPARATORS = ["\n\n", "\n", " ", ""]


def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=SEPARATORS
    )

    chunks = splitter.split_documents(documents=documents)

    print(f"Original length:  {len(documents[0].page_content)}")
    print(f"Number of chunks:  {len(chunks)}")
    print(f"Chunk sizes:  {[len(chunk.page_content) for chunk in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0].page_content[:200]}...")


def chunk_size_comparison():
    sizes = [200, 500, 1000]
    overlaps = [int(size * 0.2) for size in sizes]

    for size, overlap in zip(sizes, overlaps):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS
        )

        chunks = splitter.split_documents(documents=documents)

        print(f"Total chunks for size = {size}, overlap = {overlap}: {len(chunks)}")


def overlap_importance():
    CHUNK_SIZE = 500

    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0, separators=SEPARATORS
    )

    chunks_no_overlap = splitter_no_overlap.split_documents(documents=documents)

    splitter_with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=50, separators=SEPARATORS
    )

    chunks_with_overlap = splitter_with_overlap.split_documents(documents=documents)

    i = 8

    print("Without overlap:")
    print(f"\tChunk {i} end:   ...{chunks_no_overlap[i].page_content[-60:]}")
    print(f"\tChunk {i+1} start: {chunks_no_overlap[i+1].page_content[:60]}...")

    print("\nWith overlap (50):")
    print(f"\tChunk {i} end:   ...{chunks_with_overlap[i].page_content[-60:]}")
    print(f"\tChunk {i+1} start: {chunks_with_overlap[i+1].page_content[:60]}...")


def markdown_splitter():
    headers_to_consider = [("#", "h1"), ("##", "h2"), ("###", "h3")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)

    chunks = splitter.split_text(documents[0].page_content)

    print(f"Markdown Splitter produced {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---\n{chunk.page_content[:200]}...\n")
        print(f"\tMetadata: {chunk.metadata}\n")
        print(f"\tContent: {chunk.page_content[:200]}...\n")


def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )

    chunks = python_splitter.split_text(code_documents[0].page_content)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i} ({len(chunk)} chars)")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


# recursive_splitter()
# chunk_size_comparison()
# overlap_importance()
# markdown_splitter()
code_splitter()
