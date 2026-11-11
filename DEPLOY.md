# 部署说明（GitHub + Streamlit Community Cloud）

本文档说明如何把「工业数据分析 Agent 系统」推送到 GitHub 公开仓库，并一键部署到
Streamlit Community Cloud，使任何浏览器都能访问，同时确保 API 密钥不进入代码仓库。

---

## 一、部署前必须知道的 5 件事

| 项目 | 说明 |
| --- | --- |
| 唯一必填配置 | `DEEPSEEK_API_KEY`。留空也能运行（自动使用本地模板模式，图表与报告照常生成） |
| 密钥存放位置 | 本地：`config/.env`（已被 `.gitignore` 忽略）；云端：Streamlit Cloud 后台 Secrets |
| 云端文件系统 | **临时性**：`data/`、`output/` 中的上传文件、图表、报告会在应用重启/休眠后清空，属正常现象 |
| 中文字体 | 仓库已提供 `packages.txt`（`fonts-noto-cjk`），云端构建时自动安装，保证图表中文不乱码 |
| 入口文件 | `app.py`（建议把本项目目录作为仓库根目录，使 `app.py`、`requirements.txt` 位于根目录） |

> ⚠️ 安全提醒：如果 API Key 曾经出现在聊天记录、终端输出、截图或旧提交中，请先去
> DeepSeek 控制台**吊销并重新生成**，再用于部署。

---

## 二、把代码推送到 GitHub 公开仓库

### 1. 本地初始化仓库

在项目根目录（`app.py` 所在目录）执行：

```bash
git init
git add .
git commit -m "feat: 工业数据分析 Agent 系统（Streamlit 部署版）"
git branch -M main
```

### 2. 推送到 GitHub

在 GitHub 上新建一个 **Public** 仓库（不要勾选自动生成 README），然后执行：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

### 3. 提交前安全检查（务必执行）

```bash
# 方式一：确认 .env 没有被跟踪（应无输出）
git ls-files | grep -E "\.env$|config/\.env$"

# 方式二：确认 .env 处于忽略状态（应显示 config/.env）
git status --ignored --short | grep -E "\.env"

# 方式三：全仓库扫描疑似密钥（应无 sk- 开头的长串）
git grep -nE "sk-[A-Za-z0-9]{16,}"
```

`.gitignore` 已默认忽略：
`.env` / `config/.env` / `.streamlit/secrets.toml`、`venv/` 与 `.venv/`、`output/` 与
沙箱临时目录、`__pycache__` 与 `*.pyc`、`.idea/` / `.vscode/`、日志与本地打包产物。

> 若发现密钥已经被提交过：立即在控制台**吊销密钥**，并从 Git 历史中清除后强制推送
> （仅删除当前文件不足以消除历史泄露）。

---

## 三、部署到 Streamlit Community Cloud

### 1. 登录并创建应用

1. 打开 <https://share.streamlit.io>，使用 GitHub 账号登录；
2. 点击 **New app**；
3. 填写：
   - **Repository**：选择刚推送的仓库；
   - **Branch**：`main`；
   - **Main file path**：`app.py`（若仓库根目录是 `industrial_data_agent/` 的上级目录，则填 `industrial_data_agent/app.py`）；
4. 展开 **Advanced settings**：
   - **Python version**：3.11 或 3.12（两者均已验证）；
   - **Secrets**：粘贴下文的配置（见第 4 节）；
5. 点击 **Deploy!**，首次构建约 2–5 分钟。

> 说明：Streamlit Cloud 会自动读取仓库根目录（或应用文件同目录）的 `requirements.txt`
> 与 `packages.txt`。请勿在根目录放多个 `requirements.txt`。

### 2. 部署成功后的访问地址

部署完成后会得到一个形如 `https://<app-name>-<random>.streamlit.app` 的公网地址，
可直接分享给任何浏览器访问，无需安装任何软件。

---

## 四、在后台配置环境变量（重点：DeepSeek 密钥）

### 1. 打开 Secrets 配置

应用页面右上角 **⋮ → Settings → Secrets**（或应用卡片 → Edit → Secrets）。

### 2. 粘贴配置（TOML 格式）

最小配置只需要一行：

```toml
DEEPSEEK_API_KEY = "sk-你的真实密钥"
```

完整可选配置如下（不写则使用代码内默认值）：

```toml
DEEPSEEK_API_KEY = "sk-你的真实密钥"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

CODE_MODEL = "deepseek-chat"
REPORT_MODEL = "deepseek-chat"
FALLBACK_MODEL = "deepseek-v4-pro"

LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 3
CODE_TEMPERATURE = 0.0
REPORT_TEMPERATURE = 0.3

TEMPLATE_ROUTER = "true"

SANDBOX_TIMEOUT = 60
SANDBOX_MAX_OUTPUT_CHARS = 20000
SANDBOX_MEMORY_MB = 2048

APP_LOG_LEVEL = "INFO"
MAX_UPLOAD_MB = 200
```

### 3. 保存并重启

点击 **Save** 后，应用会自动重启。侧边栏应显示：

- `API Key 已配置：sk-f******dacc`（脱敏显示）
- `SDK 可用：True`、`传输通道：openai SDK`

### 4. 配置读取机制说明

项目统一通过 `os.getenv` 读取配置。应用启动时会按以下优先级装配环境变量：

**系统环境变量 > Streamlit Cloud Secrets > 本地 `config/.env`**

因此本地开发与云端部署使用完全相同的键名，只需切换配置来源：

- 本地：`cp config/.env.example config/.env` 后填写；
- 云端：在 Secrets 中填写（**不要**上传 `.env` 文件）。

`config/.env.example` 列出了全部配置项及说明，可直接作为模板。

---

## 五、代码更新后如何自动同步

Streamlit Community Cloud 默认监听所绑定分支：**只要向 `main` 分支推送提交，就会自动
拉取最新代码并重新部署**，无需手动操作。

```bash
git add .
git commit -m "fix: 调整模板匹配逻辑"
git push origin main
```

常见补充操作：

- 依赖或系统包变更后，可在应用菜单 **Reboot app** 强制重新构建；
- 页面行为异常但代码已更新：**⋮ → Clear cache** 后重新加载；
- 若部署日志报依赖冲突，检查 `requirements.txt` 是否仍在根目录且未被改动。

---

## 六、绑定自定义域名（可选）

需要说明的是：**Streamlit Community Cloud 不提供原生的自定义域名绑定**（免费版固定为
`*.streamlit.app`）。若展示时确实需要一个自己的域名，可选做法：

1. **反向代理（推荐）**：在自有服务器 / 云主机上用 Nginx 代理到 Streamlit 地址。
   Streamlit 依赖 WebSocket，必须转发 `Upgrade` / `Connection` 头，否则页面会一直
   "Connecting"：

   ```nginx
   server {
       listen 443 ssl;
       server_name data-agent.example.com;

       location / {
           proxy_pass https://<app-name>-<random>.streamlit.app;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host <app-name>-<random>.streamlit.app;
           proxy_read_timeout 86400;
       }
   }
   ```

2. **纯跳转（最简单但有局限）**：用 DNS/Cloudflare 做 `data-agent.example.com →
   *.streamlit.app` 的 301/302 跳转。地址栏会变成 Streamlit 域名，仅适合作为短链接入口。

3. 如果必须"自有域名 + 完全控制"，建议改用支持自定义域名的部署方案（如自建服务器 /
   容器平台），本项目代码无需改动即可迁移。

---

## 七、常见问题排查

| 现象 | 原因与解决 |
| --- | --- |
| 构建失败，提示 `No module named openai` | `requirements.txt` 未提交或不在根目录；确认仓库中存在该文件并重新部署 |
| 图表中文显示为方块 | `packages.txt` 未被读取；确认文件位于仓库根目录，内容为 `fonts-noto-cjk`，然后 Reboot app |
| 侧边栏显示"未配置 API Key" | Secrets 未保存、键名拼写不一致，或保存后未重启；确认 `DEEPSEEK_API_KEY` 完全一致 |
| 调用 LLM 报 400 模型不支持 | 服务端支持的模型名与 `CODE_MODEL` / `REPORT_MODEL` 不一致；在 Secrets 中改为可用模型名 |
| 上传文件或分析结果"消失" | 云端文件系统是临时的，重启/休眠后清空，属正常行为；重要结果请下载报告保存 |
| 应用休眠后首次访问很慢 | Community Cloud 免费实例闲置会休眠，首次唤醒需要几十秒 |
| 上传大文件失败 | `MAX_UPLOAD_MB` 与 Streamlit 服务端上传上限需同时满足（默认 200 MB） |
| 执行超时 | 在 Secrets 中调大 `SANDBOX_TIMEOUT`（默认 60 秒） |
| 沙箱报"内存不足" | Linux 云环境按 `SANDBOX_MEMORY_MB` 限制地址空间，可适当调大或降低数据规模 |

---

## 八、上线前最终检查清单

- [ ] `git ls-files` 中不存在 `config/.env`、`.streamlit/secrets.toml`
- [ ] 全仓库无 `sk-` 开头的真实密钥（`git grep -nE "sk-[A-Za-z0-9]{16,}"`）
- [ ] `requirements.txt`、`packages.txt` 位于仓库根目录
- [ ] Streamlit Cloud 的 Secrets 已填写 `DEEPSEEK_API_KEY`
- [ ] 公网地址可打开，侧边栏显示"API Key 已配置"
- [ ] 用示例问题跑通一次「标准场景（本地模板）」与一次「自定义场景（LLM）」
- [ ] 下载一次 Markdown / HTML 报告，确认图表正常
