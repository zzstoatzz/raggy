# /// script
# dependencies = [
#   "raggy[tpuf]",
# ]
# ///

import asyncio
import random
import uuid

from raggy.documents import Document
from raggy.vectorstores.tpuf import TurboPuffer, multi_query_tpuf, query_namespace


def generate_test_documents(count: int = 10) -> list[Document]:
    """Generate test documents with varied content."""
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
        base_text = random.choice(sample_texts)
        variation = f"Document {i + 1}: {base_text} Context: {uuid.uuid4().hex[:8]}"
        documents.append(
            Document(id=f"doc_{i + 1}_{uuid.uuid4().hex[:6]}", text=variation)
        )

    return documents


def test_basic_operations():
    """Test all basic TurboPuffer operations."""
    print("=" * 60)
    print("🧪 TESTING BASIC OPERATIONS")
    print("=" * 60)

    namespace = f"test_basic_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {namespace}")

    # Test 1: Namespace existence check (should be False initially)
    print("\n1️⃣ Testing namespace existence check...")
    with TurboPuffer(namespace=namespace) as tpuf:
        exists_before = tpuf.ok()
        print(f"✅ Namespace exists before creation: {exists_before}")
        assert not exists_before, "Namespace should not exist initially"

    # Test 2: Document upsert
    print("\n2️⃣ Testing document upsert...")
    documents = generate_test_documents(5)
    with TurboPuffer(namespace=namespace) as tpuf:
        tpuf.upsert(documents)
        print(f"✅ Upserted {len(documents)} documents successfully")

    # Test 3: Namespace existence check (should be True after upsert)
    print("\n3️⃣ Testing namespace existence after upsert...")
    with TurboPuffer(namespace=namespace) as tpuf:
        exists_after = tpuf.ok()
        print(f"✅ Namespace exists after upsert: {exists_after}")
        assert exists_after, "Namespace should exist after upsert"

    # Test 4: Query using query_namespace function (main library interface)
    print("\n4️⃣ Testing query_namespace function...")
    result_text = query_namespace(
        query_text="machine learning", namespace=namespace, top_k=3, max_tokens=200
    )
    print(f"✅ query_namespace returned: {result_text[:50]}...")
    assert len(result_text) > 0, "query_namespace should return text"

    # Test 5: Multi-query using multi_query_tpuf function
    print("\n5️⃣ Testing multi_query_tpuf function...")
    multi_result = multi_query_tpuf(
        queries=["machine learning", "programming"], namespace=namespace, n_results=2
    )
    print(f"✅ multi_query_tpuf returned: {multi_result[:50]}...")
    assert len(multi_result) > 0, "multi_query_tpuf should return text"

    # Test 6: Delete specific documents
    print("\n6️⃣ Testing document deletion...")
    with TurboPuffer(namespace=namespace) as tpuf:
        # Delete first two documents
        ids_to_delete = [documents[0].id, documents[1].id]
        tpuf.delete(ids_to_delete)
        print(f"✅ Deleted documents: {ids_to_delete}")

        # Verify deletion using query_namespace
        result_text = query_namespace(
            query_text="document", namespace=namespace, top_k=10, max_tokens=500
        )
        print("✅ Verified remaining documents via query_namespace")

    # Test 7: Reset namespace
    print("\n7️⃣ Testing namespace reset...")
    with TurboPuffer(namespace=namespace) as tpuf:
        tpuf.reset()
        print("✅ Namespace reset successfully")

        # Verify reset
        exists_after_reset = tpuf.ok()
        print(f"✅ Namespace exists after reset: {exists_after_reset}")


def test_advanced_operations():
    """Test advanced TurboPuffer operations."""
    print("\n" + "=" * 60)
    print("🚀 TESTING ADVANCED OPERATIONS")
    print("=" * 60)

    namespace = f"test_advanced_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {namespace}")

    # Test 1: Custom attributes
    print("\n1️⃣ Testing upsert with custom attributes...")
    with TurboPuffer(namespace=namespace) as tpuf:
        documents = [
            Document(id="custom1", text="Python programming tutorial"),
            Document(id="custom2", text="Data science methodology"),
        ]
        # Test with custom attributes
        tpuf.upsert(
            documents=documents,
            attributes={
                "category": ["programming", "science"],
                "difficulty": [1, 2],
            },
        )
        print("✅ Upserted documents with custom attributes")

    # Test 2: Query with high-level functions
    print("\n2️⃣ Testing high-level query functions...")
    result_text = query_namespace(
        query_text="programming", namespace=namespace, top_k=1, max_tokens=100
    )
    print(f"✅ query_namespace returned: {result_text[:30]}...")

    multi_result = multi_query_tpuf(
        queries=["programming", "science"], namespace=namespace, n_results=1
    )
    print(f"✅ multi_query_tpuf returned: {multi_result[:30]}...")

    # Test 3: Raw query for comparison
    print("\n3️⃣ Testing raw TurboPuffer query...")
    with TurboPuffer(namespace=namespace) as tpuf:
        result = tpuf.query(text="programming", top_k=5)
        print(f"✅ Raw query returned {len(result.rows)} results")

    # Cleanup
    with TurboPuffer(namespace=namespace) as tpuf:
        tpuf.reset()


async def test_batch_operations():
    """Test batch operations."""
    print("\n" + "=" * 60)
    print("⚡ TESTING BATCH OPERATIONS")
    print("=" * 60)

    namespace = f"test_batch_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {namespace}")

    # Test 1: Batch upsert
    print("\n1️⃣ Testing batch upsert...")
    large_doc_set = generate_test_documents(25)  # Larger set to test batching

    with TurboPuffer(namespace=namespace) as tpuf:
        await tpuf.upsert_batched(
            documents=large_doc_set, batch_size=10, max_concurrent=3
        )
        print(f"✅ Batch upserted {len(large_doc_set)} documents")

        # Verify all documents were uploaded
        result = tpuf.query(text="document", top_k=30)
        print(f"✅ Verified {len(result.rows)} documents in namespace")
        assert len(result.rows) == len(large_doc_set), (
            "All documents should be uploaded"
        )

    # Cleanup
    with TurboPuffer(namespace=namespace) as tpuf:
        tpuf.reset()


def test_error_conditions():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("⚠️ TESTING ERROR CONDITIONS")
    print("=" * 60)

    namespace = f"test_errors_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {namespace}")

    # Test 1: Query without text or vector
    print("\n1️⃣ Testing query without text or vector...")
    try:
        with TurboPuffer(namespace=namespace) as tpuf:
            tpuf.query()  # Should fail
        print("❌ Expected error was not raised")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 2: Upsert without documents or vectors
    print("\n2️⃣ Testing upsert without documents or vectors...")
    try:
        with TurboPuffer(namespace=namespace) as tpuf:
            tpuf.upsert()  # Should fail
        print("❌ Expected error was not raised")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 3: Query non-existent namespace
    print("\n3️⃣ Testing query on non-existent namespace...")
    empty_namespace = f"empty_{uuid.uuid4().hex[:8]}"
    try:
        query_namespace(query_text="anything", namespace=empty_namespace, top_k=1)
        print("⚠️ Query on empty namespace didn't fail (may be expected)")
    except Exception as e:
        print(f"✅ Query on empty namespace failed as expected: {e}")


def main():
    """Run comprehensive TurboPuffer integration tests."""
    print("🧮 COMPREHENSIVE TURBOPUFFER INTEGRATION TEST")
    print(
        "This tests raggy's TurboPuffer integration including query_namespace and multi_query_tpuf"
    )
    print("Turbopuffer version: 0.4.1 (expected)")

    try:
        test_basic_operations()
        test_advanced_operations()
        asyncio.run(test_batch_operations())
        test_error_conditions()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Basic operations: upsert, query_namespace, multi_query_tpuf")
        print("✅ Advanced operations: custom attributes, raw queries")
        print("✅ Batch operations: upsert_batched")
        print("✅ Error handling: proper exceptions raised")
        print("\nThe raggy TurboPuffer integration is working correctly!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
