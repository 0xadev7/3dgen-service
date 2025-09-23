import requests, time

def main():
    prompt = "a glossy blue sports car, studio lighting"
    t0 = time.time()
    r = requests.post("http://127.0.0.1:8000/generate/", data={"prompt": prompt})
    assert r.status_code == 200, r.text
    print("PLY bytes:", len(r.content))
    t1 = time.time()
    print("Generate took", round(t1-t0,2), "s")

    r2 = requests.post("http://127.0.0.1:8000/preview_png/", data={"prompt": prompt})
    assert r2.status_code == 200
    with open("preview.png", "wb") as f:
        f.write(r2.content)
    print("Preview saved to preview.png")

if __name__ == "__main__":
    main()
