from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import os
import argparse
from ase.io import read, write
from schnet import md

# This file serves to make a basic frontend made using Streamlit

app = FastAPI()

def return_file(path):
    return FileResponse(path)  

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename}

@app.get("/mdpredictor/{item_id}")
def mdpred(item_id: str):
    print("python scripts/example_md_predictor.py ./models/c20/" + " " + item_id + " > output.txt")
    os.system("python scripts/example_md_predictor.py ./models/c20/" + " " + item_id + " > output.txt")
    return return_file("output.txt")