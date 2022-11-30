from typing import Union
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

def return_file(path):
    return FileResponse(path)    

@app.get("/generation/{item_id}")
def generate(item_id: str):
    print("python main.py --dataset " + item_id + " --config-file test_config.json --generation True --load_cpt ./check_points/pretrained_model.pickle")
    os.system("python main.py --dataset " + item_id + " --config-file test_config.json --generation True --load_cpt ./check_points/pretrained_model.pickle")
    return return_file("/generated_samples/generated_smiles.smi")
    
@app.get("/evaluation/{item_id}")
def evaluate(item_id: str):
    os.system("python analysis/evaluate_generated_mols.py ZINC generated_samples/generated_smiles.smi zinc/smi_train.txt 1 True None analysis/wehi_pains.csv > output.txt")
    return return_file("output.txt")

