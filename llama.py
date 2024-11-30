from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import uvicorn
from typing import Optional, Dict

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    model_name: Optional[str] = "llama2"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500

class LLMManager:
    def __init__(self):
        self.llm_cache: Dict[str, Ollama] = {}
        self.base_url = "http://localhost:11434"
    
    def get_llm(self, model_name: str, temperature: float) -> Ollama:
        """Get or create LLM instance"""
        cache_key = f"{model_name}_{temperature}"
        if cache_key not in self.llm_cache:
            try:
                self.llm_cache[cache_key] = Ollama(
                    model=model_name,
                    base_url=self.base_url,
                    temperature=temperature
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize LLM: {str(e)}"
                )
        return self.llm_cache[cache_key]

llm_manager = LLMManager()

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "llama-api"}

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            "llama2",
            "mistral",
            "codellama",
            "llama2-uncensored"
        ]
    }

@app.post("/generate")
async def generate_response(request: QueryRequest):
    """Generate LLM response"""
    try:
        llm = llm_manager.get_llm(request.model_name, request.temperature)
        
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "query"]
        )
        
        chain = LLMChain(llm=llm, prompt=PROMPT)
        
        response = chain.run(
            context=request.context,
            query=request.query
        )
        
        return {
            "status": "success",
            "response": response,
            "model": request.model_name
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/chat")
async def chat_completion(request: QueryRequest):
    """Chat completion endpoint"""
    try:
        llm = llm_manager.get_llm(request.model_name, request.temperature)
        
        response = llm(request.query)
        
        return {
            "status": "success",
            "response": response,
            "model": request.model_name
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in chat completion: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
