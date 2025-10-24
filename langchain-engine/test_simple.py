"""
Simple test script for RAG system
Run this to verify everything works before using the API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("🧪 TESTING LANGCHAIN RAG SYSTEM")
print("=" * 60)

# Test 1: Environment variables
print("\n1️⃣ Testing environment variables...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key != "TU_API_KEY_AQUÍ":
    print("   ✅ OpenAI API key found")
else:
    print("   ❌ OpenAI API key not configured")
    print("   Please edit .env file and add your API key")
    exit(1)

# Test 2: Imports
print("\n2️⃣ Testing imports...")
try:
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    print("   ✅ All imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 3: Load document
print("\n3️⃣ Testing document loading...")
try:
    loader = TextLoader("./data/sample_docs/solar_energy_basics.txt")
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} document(s)")
    print(f"   📄 Content length: {len(documents[0].page_content)} characters")
except Exception as e:
    print(f"   ❌ Error loading document: {e}")
    exit(1)

# Test 4: Text splitting
print("\n4️⃣ Testing text splitting...")
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")
    print(f"   📝 First chunk preview: {chunks[0].page_content[:100]}...")
except Exception as e:
    print(f"   ❌ Error splitting text: {e}")
    exit(1)

# Test 5: Embeddings
print("\n5️⃣ Testing embeddings generation...")
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    print("   ✅ Embeddings model initialized")
except Exception as e:
    print(f"   ❌ Error initializing embeddings: {e}")
    exit(1)

# Test 6: Vector store
print("\n6️⃣ Testing vector store creation...")
try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_test"
    )
    print("   ✅ Vector store created")
except Exception as e:
    print(f"   ❌ Error creating vector store: {e}")
    exit(1)

# Test 7: Similarity search
print("\n7️⃣ Testing similarity search...")
try:
    query = "What is the optimal tilt angle for solar panels in Spain?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"   ✅ Found {len(results)} relevant chunks")
    print(f"   📌 Most relevant chunk:")
    print(f"      {results[0].page_content[:200]}...")
except Exception as e:
    print(f"   ❌ Error in similarity search: {e}")
    exit(1)

# Test 8: LLM query
print("\n8️⃣ Testing LLM generation...")
try:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4",
        temperature=0
    )
    
    # Create context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.predict(prompt)
    print("   ✅ LLM response generated")
    print(f"\n   🤖 Answer:\n   {response}")
    
except Exception as e:
    print(f"   ❌ Error generating response: {e}")
    exit(1)

# Success!
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n✅ Your RAG system is working correctly!")
print("✅ Next step: Start the FastAPI server")
print("\n💡 To start the API, run:")
print("   cd api")
print("   python fastapi_app.py")
print("=" * 60)