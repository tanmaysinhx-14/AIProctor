import asyncio, base64, json, cv2, websockets

async def main():
  cap = cv2.VideoCapture(0)
  async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=10_000_000) as ws:
    for _ in range(30):
      ok, frame = cap.read()
      if not ok:
        break
      _, buf = cv2.imencode(".jpg", frame)
      await ws.send(json.dumps({"frame": base64.b64encode(buf.tobytes()).decode()}))
      resp = json.loads(await ws.recv())
      active = resp.get("riskBreakdown", {}).get("activeRules", [])
      top = ", ".join(f'{r.get("code")}:+{r.get("points")}' for r in active[:2]) or "none"
      print(
        resp["faceCount"],
        resp["riskScore"],
        resp["riskLevel"],
        resp["metrics"]["fps"],
        resp["metrics"]["yoloTimeMs"],
        top,
      )
  cap.release()

asyncio.run(main())