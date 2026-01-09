Prompt Agent
============

專案概述
--------
Prompt Agent 是一個多階段提示詞優化流程，會依序執行 6 個階段
（診斷 -> 問答 -> 整合）來把輸入提示詞整理成更清楚、更完整的輸出提示詞。

快速開始（Windows）
-------------------
1) 把你的提示詞放在 `user/input_prompt.txt`
2) 執行：
   - `start.bat`（建議）
   - 或 `python main.py`
3) 最終結果輸出到 `user/output_prompt.txt`

執行過程中 CLI 會引導你完成追問。

LLM 服務需求
------------
本專案使用 OpenAI 相容 API。預設連線到 Ollama：
`http://127.0.0.1:11434/v1`

建議的 Ollama 設定：
1) 啟動服務：
   - `set OLLAMA_HOST=0.0.0.0:11434` 後執行 `ollama serve`
2) 下載模型：
   - `ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M`

環境變數
--------
可用以下環境變數覆寫預設設定：
- `LLM_BASE_URL`（例如 `http://127.0.0.1:11434/v1`）
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`

使用 Podman 執行（Containerfile）
-------------------------------
重點：容器內的 `localhost` 只會指向容器自己，請改用主機位址。

1) 建立映像：
   - `podman build -t prompt-agent -f Containerfile .`
2) 準備輸入檔（容器會讀寫 `user/` 與 `outputs/`）：
   - `user/input_prompt.txt`
3) 設定 `LLM_BASE_URL`（依你的環境擇一）：
   - `http://host.containers.internal:11434/v1`（Windows/macOS 常見）
   - `http://<主機IP>:11434/v1`（可用 `podman machine ip` 取得）
   - Linux 若無法解析 `host.containers.internal`，可加：
     `--add-host=host.containers.internal:host-gateway`
4) 執行容器（需要互動式 CLI）：
   - 範例（請替換 `<你的位址>`）：
     ```powershell
     podman run --rm -it -e LLM_BASE_URL=<你的位址> -v ${PWD}/user:/app/user -v ${PWD}/outputs:/app/outputs prompt-agent
     ```

需要保留 log 時可再加：`-v ${PWD}/logs:/app/logs`。

專案結構
--------
- `agentcore/`：我自行設計的核心框架（子模組）
- `main.py`：完整流程入口
- `start.bat`：Windows 啟動腳本（含 log）
- `user/input_prompt.txt`：輸入提示詞
- `user/output_prompt.txt`：輸出結果
- `config/json_config/`：JSON 設定檔
- `config/prompts/`：各階段提示詞
- `logs/`：執行紀錄（git 已忽略）

注意事項
--------
整個系統比我想得還要複雜，我花了些時間去處理錯誤追蹤和測試單元。
所以提示詞沒有很細的去修改和潤拭，所以體驗效果滿糟糕的。
後續有機會會再把程式碼重構和去提升提示詞，感覺這樣的東西很大的提升淺利。
Outputs裡面的輸入從四個字跑完這樣的流程可以變成還算完整的提示詞。

疑難排解
--------
- 出現連線錯誤時，請確認 LLM 服務已啟動，且 `LLM_BASE_URL` 可被連線。
- 使用 Podman 時，請確認容器能連到主機 11434 連接埠。
