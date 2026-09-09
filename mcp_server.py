import sys
import json
from client import TensorCoreWMMA

wmma = TensorCoreWMMA(4, 4, 4)

def handle_call(name, arguments):
    if name == "wmma_sync":
        a = arguments["a"]
        b = arguments["b"]
        c = arguments["c"]
        m = len(a)
        k = len(b)
        n = len(b[0])
        eng = TensorCoreWMMA(m, n, k)
        res = eng.wmma_sync(a, b, c)
        return {"result_tile": res}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
