# sonar_estimator.py
import base64
import pathlib
import requests
from aiq.cli.type_registry import type_registry

def sonar_estimator(image_path: str | pathlib.Path) -> dict:
    image_path = pathlib.Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img_b64 = base64.b64encode(image_path.read_bytes()).decode()

    resp = requests.post(
        "http://127.0.0.1:8800/estimate",
        json={"image_base64": img_b64},
        timeout=15,
    )
    resp.raise_for_status()  

    return resp.json()  

type_registry.register_tool(
    func=sonar_estimator,
    name="sonar_estimator",
    namespace="aiq.tools",
    description="呼叫聲納裝置拍攝影像並估算魚群數量..."
)