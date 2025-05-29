# /// script
# dependencies = [
#   "raggy[tpuf]",
# ]
# ///

import random
import uuid

from raggy.documents import Document
from raggy.vectorstores.tpuf import TurboPuffer, query_namespace


def generate_random_documents(count: int = 10) -> list[Document]:
    """Generate random documents for testing."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is revolutionizing the way we process data.",
        "Python is a versatile programming language for data science.",
        "Natural language processing enables computers to understand human language.",
        "Vector databases are essential for similarity search applications.",
        "Artificial intelligence is transforming various industries.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        "Open source software fosters collaboration and innovation.",
        "Data visualization helps in understanding complex datasets.",
        "Automation improves efficiency and reduces manual errors.",
        "Cybersecurity is crucial for protecting digital assets.",
        "The internet has connected people across the globe.",
        "Quantum computing promises to solve complex problems faster.",
        "Blockchain technology ensures transparency and security.",
        "Mobile applications have changed how we interact with technology.",
    ]

    documents = []
    for i in range(count):
        # Pick random text and add some variation
        base_text = random.choice(sample_texts)
        variation = (
            f" Document {i + 1}: {base_text} Additional context: {uuid.uuid4().hex[:8]}"
        )

        documents.append(Document(id=f"doc_{uuid.uuid4().hex[:8]}", text=variation))

    return documents


def main():
    """Test Turbo Puffer integration by writing and querying random documents."""
    # Generate random namespace
    namespace = f"test_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {namespace}")

    # Generate random documents
    documents = generate_random_documents(15)
    print(f"Generated {len(documents)} random documents")

    # Write to Turbo Puffer
    with TurboPuffer(namespace=namespace) as tpuf:
        print("Writing documents to Turbo Puffer...")
        tpuf.upsert(documents)
        print("✅ Documents written successfully!")

        # Test if the namespace exists
        if tpuf.ok():
            print("✅ Namespace exists and is accessible")
        else:
            print("❌ Namespace check failed")

    # Test querying
    print("\nTesting queries...")
    test_queries = [
        "machine learning",
        "programming language",
        "data science",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = query_namespace(query, namespace=namespace, top_k=3)
            print(f"Results: {results[:200]}...")
        except Exception as e:
            print(f"Query failed: {e}")

    # Cleanup
    print(f"\nCleaning up namespace: {namespace}")
    with TurboPuffer(namespace=namespace) as tpuf:
        tpuf.reset()
    print("✅ Cleanup complete!")


if __name__ == "__main__":
    main()
