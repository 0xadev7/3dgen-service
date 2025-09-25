import os, json, time
from app.serverless import handler

def main():
    event = {"input": {"prompt": "a simple red cube, product photo", "steps": 2, "n_points": 10000, "return_ply_b64": False}}
    t0 = time.time()
    res = handler(event)
    assert res.get("ok"), f"Sanity failed: {res}"
    assert os.path.exists(res["ply_path"]), "PLY not written"
    print(json.dumps({"status":"ok","latency_ms": int((time.time()-t0)*1000), "ply_path": res["ply_path"]}))
if __name__ == "__main__":
    main()
