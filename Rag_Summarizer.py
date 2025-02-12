

import streamlit as st
import os
import tempfile
import re
import arxiv
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ------------------------
# 1. Custom CSS & Configuration
# ------------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        .debug-info {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .document-chunk {
            border: 1px solid #ddd;
            padding: 10px;
            margin: 5px 0;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@dataclass
class ExperimentConfig:
    model: str
    temperature: float
    max_tokens: int
    # chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    custom_separators: Optional[str]
    prompt_style: str
    custom_prompt: Optional[str]
    evaluation_metrics: List[str]

# ------------------------
# 2. Document Processing Helpers
# ------------------------
def split_document(doc: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a document into smaller chunks using a RecursiveCharacterTextSplitter.
    Returns a list of Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(doc.page_content)
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
        new_doc.metadata["chunk_id"] = i
        chunked_docs.append(new_doc)
    return chunked_docs

def process_uploaded_files(uploaded_files, config: ExperimentConfig) -> List[Document]:
    """
    Processes uploaded files: saves them temporarily, loads content via an unstructured loader,
    and splits them if needed. Returns a list of Document objects.
    """
    all_docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.read())
                st.write(f"Processing file: {uploaded_file.name}")
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                for idx, doc in enumerate(docs):
                    # Use the file name as the title (or modify as needed)
                    doc.metadata.update({
                        "source": uploaded_file.name,
                        "title": uploaded_file.name,
                        "type": "upload",
                        "file_type": uploaded_file.type,
                        "doc_id": f"{uploaded_file.name}_{idx}",
                        "content_length": len(doc.page_content),
                    })
                    # Split if content is too long
                    if len(doc.page_content) > config.chunk_size:
                        chunks = split_document(doc, config.chunk_size, config.chunk_overlap)
                        all_docs.extend(chunks)
                    else:
                        all_docs.append(doc)
                st.write(f"Processed file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return all_docs

def build_vector_store(documents: List[Document]) -> FAISS:
    """
    Builds a FAISS vector store from the provided documents.
    Returns the vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

def aggregate_docs_by_title(docs: List[Document]) -> Dict[str, str]:
    """
    Aggregates document chunks by their title.
    Returns a dictionary mapping each title to concatenated content.
    """
    agg = defaultdict(list)
    for doc in docs:
        agg[doc.metadata.get("title", "Unknown")].append(doc.page_content)
    # Join the chunks with a newline separator
    return {title: "\n".join(contents) for title, contents in agg.items()}

# ------------------------
# 3. Tools (with proper docstrings)
# ------------------------
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieves relevant document chunks from the vector store for the given query.
    Returns a tuple of (serialized content, list of Document objects).
    """
    if "vectorstore" not in st.session_state:
        return "No vector store available.", []
    try:
        docs = st.session_state.vectorstore.similarity_search(query, k=5)
        if not docs:
            return "Sorry, no relevant documents found in your uploaded files.", []
        serialized = "\n\n---\n\n".join(
            f"Document [{doc.metadata.get('doc_id', 'N/A')}]\n"
            f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            f"Content:\n{doc.page_content}"
            for doc in docs
        )
        return serialized, docs
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return str(e), []

# @tool
# def analyze_citations(text: str) -> List[Dict[str, str]]:
#     """
#     Extracts unique bracket citations (e.g., [1], [2]) from the given text.
#     Maps citation numbers to document metadata based on the order in st.session_state.documents.
#     Returns a list of citation dictionaries.
#     """
#     st.write("Analyzing citations in text")
#     citations = {}
#     pattern = r'\[(\d+)\]'
#     for match in re.finditer(pattern, text):
#         num_str = match.group(1)
#         if num_str.isdigit():
#             idx = int(num_str) - 1
#             if 0 <= idx < len(st.session_state.documents):
#                 doc = st.session_state.documents[idx]
#                 if num_str not in citations:
#                     citations[num_str] = {
#                         "id": f"[{num_str}]",
#                         "title": doc.metadata.get("title", "Unknown"),
#                         "authors": doc.metadata.get("authors", ["Unknown"]),
#                         "url": doc.metadata.get("url", None),
#                         "published": doc.metadata.get("published", "Unknown"),
#                     }
#                     st.write(f"Found citation: {citations[num_str]}")
#     return list(citations.values())

@tool
def summarize(text: str) -> str:
    """
    Generates a summary/analysis for the given text.
    The prompt instructs the model to include bracket citations (e.g., [1], [2])
    and end with a single References section.
    Returns the generated analysis as text.
    """
    prompt = f"""You are a research assistant. Analyze the following text and provide a detailed summary.
Make sure to label document references with bracket citations like [1], [2], etc.
End your response with a single References section listing these citations.

Text to analyze:
{text}
"""
    try:
        response = st.session_state.current_llm.invoke([SystemMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return f"Error: {str(e)}"

# ------------------------
# 4. Graph Functions
# ------------------------
def query_or_respond(state: MessagesState):
    """
    Handles a query by combining a summary message (instead of full content if too many docs)
    with the conversation messages, then invokes the model using retrieval tools.
    Returns the model's response.
    """
    if not st.session_state.documents:
        return {"messages": [AIMessage(content="No documents available.")]}
    # If too many documents, do not include all text to avoid token overflow
    if len(st.session_state.documents) > 10:
        doc_content = f"{len(st.session_state.documents)} documents loaded. Use the retrieval tool for specific queries."
    else:
        doc_content = "\n\n".join(
            f"Document [{doc.metadata.get('doc_id', 'N/A')}]:\n{doc.page_content}"
            for doc in st.session_state.documents
        )
    system_message = SystemMessage(content=f"""You are a research assistant with the following documents:

{doc_content}

Always cite documents with bracket references like [1], [2], etc.
""")
    messages = [system_message] + state["messages"]
    tools = [retrieve, summarize]
    llm_with_tools = st.session_state.current_llm.bind_tools(tools)
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

def generate(state: MessagesState):
    """
    Final response generation node.
    Combines a system message (with reduced content if too many docs) and conversation messages,
    invokes the model, and appends a single References section based on bracket citations.
    Returns the model's response.
    """
    if not st.session_state.documents:
        return {"messages": [AIMessage(content="No documents available.")]}
    if len(st.session_state.documents) > 10:
        doc_content = f"{len(st.session_state.documents)} documents loaded."
    else:
        doc_content = "\n\n".join(
            f"Document [{doc.metadata.get('doc_id', 'N/A')}]:\n{doc.page_content}"
            for doc in st.session_state.documents
        )
    system_message = f"""You are a research assistant with the following documents:

{doc_content}

Always cite documents with bracket references like [1], [2], etc.
"""
    conversation_messages = [
        msg for msg in state["messages"]
        if msg.__class__.__name__.lower() in ("humanmessage", "systemmessage") or
           (msg.__class__.__name__.lower() == "aimessage" and not getattr(msg, "tool_calls", None))
    ]
    prompt = [SystemMessage(content=system_message)] + conversation_messages
    response = st.session_state.current_llm.invoke(prompt)
    # citations = analyze_citations(response.content)
    # if citations:
    #     response.content += "\n\nReferences:\n" + "\n".join(
    #         f"{cite['id']} {cite['title']} - {', '.join(cite['authors'])}"
    #         for cite in citations
    #     )
    return {"messages": [response]}

def setup_graph():
    """
    Sets up the processing graph with nodes for query handling, tool execution, and final response generation.
    Returns the compiled graph.
    """
    graph = StateGraph(MessagesState)
    graph.add_node("query_or_respond", query_or_respond)
    tools_node = ToolNode([retrieve, summarize])
    graph.add_node("tools", tools_node)
    graph.add_node("generate", generate)
    graph.set_entry_point("query_or_respond")
    graph.add_conditional_edges(
        "query_or_respond",
        lambda x: "tools" if getattr(x["messages"][-1], "tool_calls", False) else "generate",
        {"tools": "tools", "generate": "generate"}
    )
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=MemorySaver())

# ------------------------
# 5. arXiv Functions
# ------------------------
def fetch_arxiv_papers(topic: str, num_papers: int, config: ExperimentConfig) -> List[Document]:
    """
    Searches arXiv for papers matching the topic and returns a list of Document objects.
    Splits content if it exceeds the chunk size.
    """
    try:
        st.write(f"Searching arXiv for: {topic}")
        search = arxiv.Search(
            query=topic,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )
        docs = []
        for i, result in enumerate(search.results()):
            content = f"""TITLE: {result.title}

AUTHORS: {', '.join([author.name for author in result.authors])}
DATE: {result.published.strftime('%Y-%m-%d')}
ABSTRACT:
{result.summary}

LINKS:
arXiv URL: {result.entry_id}
PDF URL: {result.pdf_url}
"""
            doc = Document(
                page_content=content,
                metadata={
                    "source": "arXiv",
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "arxiv_url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "type": "arXiv",
                }
            )
            if len(doc.page_content) > config.chunk_size:
                docs.extend(split_document(doc, config.chunk_size, config.chunk_overlap))
            else:
                docs.append(doc)
        st.success(f"Retrieved {len(docs)} papers from arXiv")
        return docs
    except Exception as e:
        st.error(f"Error fetching arXiv papers: {str(e)}")
        return []

def generate_comparative_analysis(docs: List[Document]) -> str:
    """
    Generates a comparative analysis for the provided documents.
    The prompt instructs the model to compare and contrast the documents using bracket citations.
    Returns the generated analysis.
    """
    # Aggregate content by title to reduce token usage
    aggregated = aggregate_docs_by_title(docs)
    combined_text = "\n\n".join(aggregated.values())
    prompt = f"""You are a research assistant. Provide a comparative analysis of the following documents.
- Briefly summarize each document.
- Compare and contrast their methods and findings.
- Use bracket citations like [1], [2], etc.
- End with a single References section.

Documents:
{combined_text}
"""
    response = st.session_state.current_llm.invoke([SystemMessage(content=prompt)])
    return response.content

def generate_individual_analysis(docs: List[Document]) -> str:
    """
    Generates an individual analysis for each provided document.
    Aggregates document chunks by title and instructs the model to include bracket citations.
    Returns the generated analysis.
    """
    aggregated = aggregate_docs_by_title(docs)
    combined_text = "\n\n".join(aggregated.values())
    prompt = f"""You are a research assistant. For each of the following documents, provide an individual analysis.
Include bracket citations (e.g., [1], [2]) for references.
End with a single References section.

Documents:
{combined_text}
"""
    response = st.session_state.current_llm.invoke([SystemMessage(content=prompt)])
    return response.content

# ------------------------
# 6. Analysis Interface
# ------------------------
def show_paper_analysis_interface(papers: List[Document]):
    """
    Displays the paper analysis interface.
    De-duplicates papers by title for selection and aggregates their content.
    Allows the user to choose between Individual and Comparative Analysis.
    """
    st.subheader("Paper Analysis")
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        options=["Individual Analysis", "Comparative Analysis"],
        help="Select the type of analysis."
    )
    unique_titles = sorted({doc.metadata["title"] for doc in papers})
    selected_titles = st.multiselect(
        "Select Papers to Analyze",
        options=unique_titles,
        default=unique_titles[:1] if unique_titles else [],
        help="Select one or more paper titles."
    )
    if st.button("Run Analysis", key="analysis_button"):
        if not selected_titles:
            st.warning("Please select at least one paper.")
            return
        # Filter documents and aggregate content by title
        selected_docs = [doc for doc in papers if doc.metadata["title"] in selected_titles]
        with st.spinner("Generating analysis..."):
            if analysis_type == "Comparative Analysis":
                if len(selected_titles) < 2:
                    st.warning("Select at least 2 different papers for comparative analysis.")
                    return
                analysis_text = generate_comparative_analysis(selected_docs)
            else:
                analysis_text = generate_individual_analysis(selected_docs)
            # citations = analyze_citations(analysis_text)
            # if citations:
            #     analysis_text += "\n\nReferences:\n" + "\n".join(
            #         f"{cite['id']} {cite['title']} - {', '.join(cite['authors'])}"
            #         for cite in citations
            #     )
            # st.markdown(analysis_text)

def show_citation_network(papers: List[Document]):
    """
    Displays a simple citation network based on unique paper titles.
    """
    st.subheader("Citation Network")
    unique_titles = sorted({doc.metadata["title"] for doc in papers})
    for title in unique_titles:
        doc_list = [doc for doc in papers if doc.metadata["title"] == title]
        if not doc_list:
            continue
        doc_example = doc_list[0]
        with st.expander(title, expanded=False):
            st.markdown(f"""
            **Authors:** {', '.join(doc_example.metadata.get('authors', []))}
            **Published:** {doc_example.metadata.get('published', 'Unknown')}
            [View Paper]({doc_example.metadata.get('arxiv_url', '#')})
            """)

def add_analysis_tabs():
    """
    Adds tabs for Analysis, Citations, and Trends.
    """
    tab_analysis, tab_citations, tab_trends = st.tabs(["Analysis", "Citations", "Trends"])
    with tab_analysis:
        if st.session_state.documents:
            show_paper_analysis_interface(st.session_state.documents)
        else:
            st.info("Please process documents first.")
    with tab_citations:
        if st.session_state.documents:
            show_citation_network(st.session_state.documents)
        else:
            st.info("Please process documents first.")
    with tab_trends:
        st.markdown("### Research Trends")
        if st.session_state.documents:
            dates = [doc.metadata.get("published", "Unknown") for doc in st.session_state.documents]
            st.write("Publication Timeline:")
            for date in sorted(dates):
                st.write(f"- {date}")
        else:
            st.info("Please process documents first.")

# ------------------------
# 7. Sidebar Configuration
# ------------------------
def create_experimental_sidebar() -> ExperimentConfig:
    """
    Creates an ExperimentConfig object based on sidebar selections.
    """
    st.sidebar.title("Configuration")
    with st.sidebar.expander("Model & Parameters", expanded=True):
        selected_model = st.selectbox(
            "Choose LLM Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama3-8b-8192",
                "llama-guard-3-8b",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            help="Select the language model."
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 4000, 2000, 100)
    with st.sidebar.expander("Document Processing", expanded=False):
        # chunking_strategy = st.selectbox(
        #     "Processing Strategy",
        #     options=["Complete Document", "Simple", "Custom"],
        #     help="Select how to process documents."
        # )
        chunk_size = st.number_input("Maximum Content Length", 100, 6000, 2000)
        chunk_overlap = st.number_input("Content Overlap", 0, 1000, 200)
        custom_separators = None
    prompt_style = "Detailed Analysis"
    custom_prompt = None
    evaluation_metrics = ["Response Accuracy", "Content Coverage"]
    return ExperimentConfig(
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
        # chunking_strategy=0,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        custom_separators=custom_separators,
        prompt_style=prompt_style,
        custom_prompt=custom_prompt,
        evaluation_metrics=evaluation_metrics
    )

# ------------------------
# 8. Main Application
# ------------------------
def main():
    """
    Main application function.
    Sets up the Streamlit UI, processes documents, and provides chat and analysis functionality.
    """
    st.set_page_config(page_title="Research Assistant", layout="wide")
    add_custom_css()
    st.title("Research Paper Analysis Assistant")
    
    # Initialize session state variables if not already present.
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_llm" not in st.session_state:
        st.session_state.current_llm = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    
    config = create_experimental_sidebar()
    
    # Initialize LLM if not already initialized.
    if st.session_state.current_llm is None:
        try:
            st.session_state.current_llm = ChatGroq(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            st.session_state.graph = setup_graph()
            st.success("Model initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return
    
    # Document Processing Section
    st.subheader("Document Processing")
    tab_upload, tab_arxiv = st.tabs(["Upload Files", "Search arXiv"])
    with tab_upload:
        uploaded_files = st.file_uploader("Upload Research Papers", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    with tab_arxiv:
        col1, col2 = st.columns([3, 1])
        with col1:
            arxiv_query = st.text_input("Search arXiv Papers", placeholder="e.g., 'transformer architecture'")
        with col2:
            max_papers = st.number_input("Max Papers", 1, 10, 3)
    if st.button("Process Documents", key="process_documents"):
        with st.spinner("Processing documents..."):
            # Do not clear st.session_state.documents so that documents persist.
            if uploaded_files:
                docs = process_uploaded_files(uploaded_files, config)
                st.session_state.documents.extend(docs)
                st.write(f"Processed {len(docs)} document(s) from upload.")
            if arxiv_query.strip():
                arxiv_docs = fetch_arxiv_papers(arxiv_query, max_papers, config)
                st.session_state.documents.extend(arxiv_docs)
                st.write(f"Fetched {len(arxiv_docs)} document(s) from arXiv.")
            if st.session_state.documents:
                st.session_state.vectorstore = build_vector_store(st.session_state.documents)
                st.success(f"Total documents in memory: {len(st.session_state.documents)}")
            else:
                st.warning("No documents were processed.")
    
    # Chat Interface Section
    st.subheader("Ask Questions")
    tab_chat, tab_analyze = st.tabs(["Chat", "Analyze"])
    with tab_chat:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if query := st.chat_input("Ask about the research papers..."):
            if not st.session_state.documents:
                st.error("No documents available. Please process documents first.")
                return
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Processing query..."):
                    try:
                        system_content = f"""You are a research assistant with {len(st.session_state.documents)} documents.
Always cite them with bracket references like [1], [2], etc.
"""
                        initial_state = {
                            "messages": [
                                SystemMessage(content=system_content),
                                HumanMessage(content=query)
                            ]
                        }
                        result_state = st.session_state.graph.invoke(
                            initial_state, config={"configurable": {"thread_id": "research_assistant"}}
                        )
                        if result_state.get("messages"):
                            response = result_state["messages"][-1].content
                            st.write(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    with tab_analyze:
        st.write("Generate analysis from selected documents.")
        add_analysis_tabs()

if __name__ == "__main__":
    main()