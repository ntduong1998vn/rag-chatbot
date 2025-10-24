from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello from chatbot!"}


def main():
    print("Hello from chatbot!")


if __name__ == "__main__":
    main()
