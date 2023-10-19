from fastapi import FastAPI, HTTPException,BackgroundTasks, Depends, status,Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from mocap_if import *
from bll import *

app = FastAPI()
origins = ["*"]  # Adjust this according to your frontend's URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Example token, replace it with your own authentication logic
TOKEN = "we_are_winners_2023_6789"


def verify_token(api_token: str = Header(...)):
    if api_token != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return True

# Example Models
class SearchModal(BaseModel):
    text: str

class IndexModal(BaseModel):
    ImgURL: str

# Example routes
@app.post("/api/search")
async def searchPrompt(item: SearchModal, token_valid: bool = Depends(verify_token)):
    return await search_with_text(item.text)


@app.post("/api/add_resource")
async def method2(items: List[IndexModal], token_valid: bool = Depends(verify_token)):
    return await indexFiles(items)

@app.post("/api/check_exists")
async def method2(item:IndexModal, token_valid: bool = Depends(verify_token)):
    return {"item_received": item}

# run with cmd
# uvicorn main:app  --reload --host 0.0.0.0 --port 6158
