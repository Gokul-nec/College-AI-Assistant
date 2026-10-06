import os
from typing import Tuple, List, Dict
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class LLMProvider:
    """Abstract LLM provider interface"""
    
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq LLM Provider"""
    
    def __init__(self):
        try:
            from groq import Groq
            self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            self.model = os.getenv('LLM_MODEL', 'mixtral-8x7b-32768')
        except ImportError:
            raise ImportError("Groq library not installed. Install with: pip install groq")
    
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider"""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
            self.client = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))
        except ImportError:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
    
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.generate_content(
            full_prompt,
            generation_config={
                'max_output_tokens': max_tokens,
                'temperature': temperature
            }
        )
        return response.text


class CollegeChatbot:
    """Enhanced College AI Chatbot with Multi-LLM support and RAG"""
    
    def __init__(self, persist_directory='vector_database',
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.loaded_documents = {}
        
        # Initialize LLM provider
        self.llm_provider = self._init_llm_provider()
        
        # College content categories
        self.content_categories = {
            'admissions': r'admission|apply|application|eligibility|requirement|deadline|acceptance|enroll',
            'courses': r'course|subject|curriculum|program|major|minor|credit|semester|syllabus|module',
            'fees': r'fee|tuition|payment|scholarship|grant|financial|cost|expense|aid|waiver|refund',
            'academics': r'academic|grade|gpa|transcript|exam|test|assessment|result|mark|evaluation',
            'campus': r'campus|facility|library|lab|laboratory|dorm|dormitory|building|hostel|clinic|cafeteria',
            'sports': r'sport|athletics|gym|physical|recreation|team|match|competition|tournament|coach',
            'clubs': r'club|organization|society|student|group|activity|event|cultural|committee',
            'policies': r'policy|rule|regulation|conduct|discipline|code|guideline|procedure|conduct',
            'faculty': r'faculty|professor|instructor|teacher|staff|department|contact|office|hour',
            'general': r'general|information|help|question|what|how|when|where|who|why'
        }
        
        self.load_existing_vector_store()
    
    def _init_llm_provider(self) -> LLMProvider:
        """Initialize LLM provider based on configuration"""
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        
        try:
            if provider == 'groq':
                return GroqProvider()
            elif provider == 'gemini':
                return GeminiProvider()
            else:  # Default to OpenAI
                return OpenAIProvider()
        except Exception as e:
            print(f"Warning: Could not initialize {provider} provider: {e}")
            print("Falling back to OpenAI provider...")
            try:
                return OpenAIProvider()
            except:
                raise Exception("No valid LLM provider configured. Check your .env file.")
    
    def load_existing_vector_store(self):
        """Load existing vector store if available"""
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                print(f'✓ Loaded vector store from {self.persist_directory}')
        except Exception as e:
            print(f'Warning: Could not load vector store: {e}')
    
    def load_document(self, file_path: str) -> Document:
        """Load a document from PDF or TXT file"""
        if not os.path.exists(file_path):
            return None
        
        try:
            file_ext = file_path.rsplit('.', 1)[1].lower()
            
            if file_ext == 'pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_ext == 'txt':
                loader = TextLoader(file_path)
                docs = loader.load()
            else:
                return None
            
            if not docs:
                return None
            
            content = '\n\n'.join(doc.page_content for doc in docs)
            if not content.strip():
                return None
            
            document = Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'filename': os.path.basename(file_path),
                    'file_type': file_ext,
                    'loaded_at': datetime.now().isoformat(),
                    'char_count': len(content)
                }
            )
            
            self.loaded_documents[os.path.basename(file_path)] = document
            return document
        except Exception as e:
            print(f'Error loading {file_path}: {e}')
            return None
    
    def identify_chunk_category(self, content: str) -> str:
        """Identify content category using regex patterns"""
        content_lower = content.lower()
        score = {}
        
        for category, pattern in self.content_categories.items():
            matches = len(re.findall(pattern, content_lower))
            if matches > 0:
                score[category] = matches
        
        return max(score, key=score.get) if score else 'general'
    
    def create_chunk_context(self, category: str) -> str:
        """Create context description for a category"""
        descriptions = {
            'admissions': 'Information about admissions, applications, and eligibility criteria.',
            'courses': 'Course offerings, curriculum details, and academic programs.',
            'fees': 'Fee structure, payment information, and financial aid options.',
            'academics': 'Academic policies, grading systems, and assessment methods.',
            'campus': 'Campus facilities, buildings, and infrastructure.',
            'sports': 'Sports programs, facilities, and athletic activities.',
            'clubs': 'Student clubs, organizations, and extracurricular activities.',
            'policies': 'College policies, regulations, and procedures.',
            'faculty': 'Faculty information, departments, and contact details.',
            'general': 'General college information and frequently asked questions.'
        }
        return descriptions.get(category, 'General information')
    
    def process_and_store_documents(self, file_paths: List[str]) -> int:
        """Process and store documents in vector database"""
        all_chunks = []
        
        for file_path in file_paths:
            doc = self.load_document(file_path)
            if doc:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )
                chunks = splitter.split_documents([doc])
                
                for chunk in chunks:
                    category = self.identify_chunk_category(chunk.page_content)
                    chunk.metadata.update({
                        'category': category,
                        'category_context': self.create_chunk_context(category),
                        'chunk_size': len(chunk.page_content)
                    })
                
                all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
        
        try:
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vector_store.add_documents(all_chunks)
            
            print(f'✓ Stored {len(all_chunks)} chunks in vector database')
            return len(all_chunks)
        except Exception as e:
            print(f'Error storing documents: {e}')
            return 0
    
    def search_information(self, query: str, k: int = 4):
        """Search for relevant information in vector store"""
        if not self.vector_store:
            return []
        
        try:
            enhanced_query = f'College information query: {query}'
            return self.vector_store.similarity_search_with_score(enhanced_query, k=k)
        except Exception as e:
            print(f'Error searching: {e}')
            return []
    
    def get_response(self, query: str) -> Tuple[str, List[Dict]]:
        """Generate response using RAG and LLM"""
        results = self.search_information(query, k=4)
        
        if not results:
            return (
                "I don't have specific information about that in my knowledge base. "
                "Please upload relevant college documents or try rephrasing your question.",
                []
            )
        
        context_parts = []
        sources = []
        categories_found = set()
        
        for result, score in results:
            category = result.metadata.get('category', 'general')
            categories_found.add(category)
            context_parts.append(result.page_content)
            sources.append({
                'filename': result.metadata.get('filename', 'Unknown'),
                'category': category,
                'relevance_score': float(score)
            })
        
        context = '\n\n'.join(context_parts)
        
        try:
            system_prompt = f"""You are an intelligent and helpful College AI Assistant for a prestigious educational institution.
Your role is to provide accurate, professional, and supportive responses to students, parents, and prospective students.

Core Guidelines:
- Answer ONLY using the provided college information
- Be helpful, friendly, and professional
- Structure responses clearly with bullet points when helpful
- If specific information isn't available, clearly state that
- Always maintain a positive and supportive tone
- Suggest related topics the user might be interested in
- Keep responses concise but comprehensive
- Be encouraging and welcoming to all users"""
            
            user_prompt = f"""User Question: \"{query}\"

Relevant College Information:
{context}

Please provide a helpful, accurate response based on the college information provided above. Be encouraging and thorough."""
            
            max_tokens = int(os.getenv('MAX_TOKENS', 1000))
            temperature = float(os.getenv('TEMPERATURE', 0.5))
            
            answer = self.llm_provider.generate_response(
                system_prompt,
                user_prompt,
                max_tokens,
                temperature
            )
            
            # Add topic coverage footer
            if categories_found:
                answer += f"\n\n**📚 Topics Covered:** {', '.join(sorted(categories_found))}"
            
            return answer, sources
        
        except Exception as e:
            print(f'Error generating response: {e}')
            return f"I encountered an error processing your question: {str(e)}", sources
    
    def get_vector_store_stats(self) -> Dict:
        """Get vector store statistics"""
        if not self.vector_store:
            return {'status': 'No vector store loaded'}
        
        try:
            collection = self.vector_store._collection
            return {
                'status': 'Active',
                'total_chunks': collection.count(),
                'documents': list(self.loaded_documents.keys())
            }
        except Exception as e:
            return {'status': f'Error: {e}'}


if __name__ == '__main__':
    bot = CollegeChatbot()
    print(bot.get_vector_store_stats())
