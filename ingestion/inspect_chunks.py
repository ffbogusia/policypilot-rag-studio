import argparse

from ingestion.chunk_documents import chunk_documents
from ingestion.load_documents import load_markdown_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect chunks created from policy documents.")
    parser.add_argument(
        "--docs",
        default="data/sample_policies",
        help="Path to the folder with Markdown policy documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of chunks to print.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Chunk overlap in characters for long sections.",
    )

    args = parser.parse_args()

    documents = load_markdown_documents(args.docs)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[: args.limit]:
        print()
        print(f"- chunk_id: {chunk.chunk_id}")
        print(f"  title: {chunk.title}")
        print(f"  heading: {chunk.heading}")
        print(f"  category: {chunk.category}")
        print(f"  source_path: {chunk.source_path}")
        print(f"  chars: {len(chunk.text)}")
        print(f"  preview: {chunk.preview(120)}")


if __name__ == "__main__":
    main()