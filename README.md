# Assignment : FastAPI and LangChain Integration

## Description
This assignment demonstrates the integration of FastAPI with LangChain and other libraries to build a simple AI ChatBot. It includes features such as flow mode, RAG mode and vector database interaction using Milvus.

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package manager)
- Docker (for running Milvus)

## Installation
1.Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
Additionally, here are the commands you might need:

2. Create a virtual environment:
   ```bash
   python -m venv venv
3. Activate the virtual environment:
   On Windows:
   ```bash
      venv\Scripts\activate
4. Install dependencies:
   ```bash
   pip install -r requirements.txt

5. **Run the FastAPI application**:
   ```bash
   uvicorn main:app --reload

6.**Run the Streamlit Chat Interface**:
```bash
streamlit run interface.py