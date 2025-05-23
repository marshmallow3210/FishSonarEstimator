# examples/estimator/src/estimator/estimator_function.py
import base64, pathlib, tempfile, uuid, requests
from typing import AsyncGenerator, Dict, Any

from pydantic import Field
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

class EstimatorConfig(FunctionBaseConfig, name="estimator"):
    image_path: str = Field(..., description="最新聲納影像路徑")
    api_url: str = Field("http://127.0.0.1:8800/estimate", description="Flask 推論服務 URL")

def _call_api(img_path: pathlib.Path, api_url: str) -> Dict[str, Any]:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    resp = requests.post(api_url, json={"image_base64": b64}, timeout=30)
    resp.raise_for_status()
    return resp.json()  # {"estimated_count": int, "image_base64": str}

@register_function(config_type=EstimatorConfig)
async def estimator(
    config: EstimatorConfig,
    builder: Builder
) -> AsyncGenerator[FunctionInfo, None]:

    async def run(dummy: str) -> Dict[str, Any]:
        img = pathlib.Path(config.image_path)
        if not img.exists():
            raise FileNotFoundError(img)

        result = _call_api(img, config.api_url)

        # 把 API 回傳的 base64 影像解開，存成臨時檔
        raw = base64.b64decode(result["image_base64"])
        tmp = pathlib.Path(tempfile.gettempdir()) / f"aiq_sonar_{uuid.uuid4().hex}.png"
        tmp.write_bytes(raw)

        # 回傳一個 dict：UI 會自動把 image_path 的檔案渲染成圖片
        return {
            "estimated_count": int(result["estimated_count"]),
            "image_path": str(tmp)
        }

    yield FunctionInfo.create(
        single_fn=run,
        description="回傳最新聲納影像中的魚隻數量與影像",
    )
