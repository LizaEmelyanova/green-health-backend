from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Green Health API is running!"}