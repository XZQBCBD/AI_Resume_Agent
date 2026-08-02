# 🚀 AI 数字分身系统 — 公网部署指南

> 让面试官通过网址直接访问你的 AI 数字分身！

---

## 📋 目录

1. [方案选择](#方案选择)
2. [方案一：Docker 一键部署（推荐）](#方案一docker-一键部署推荐)
3. [方案二：传统手动部署](#方案二传统手动部署)
4. [方案三：Streamlit Cloud（免费，有局限）](#方案三streamlit-cloud免费方案)
5. [域名 + HTTPS 配置](#域名--https-配置)
6. [安全注意事项](#安全注意事项)
7. [常见问题](#常见问题)

---

## 方案选择

| 方案 | 费用 | 难度 | 稳定性 | 适合场景 |
|------|------|------|--------|----------|
| **Docker + 云服务器** | ~50-100元/月 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 长期使用，面试展示 |
| 传统手动部署 | ~50-100元/月 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 熟悉 Linux 的用户 |
| Streamlit Cloud | 免费 | ⭐ | ⭐⭐ | 快速预览、临时展示 |

> 💡 **推荐方案一**：Docker 部署到阿里云/腾讯云轻量应用服务器，最稳定、最可控。

---

## 方案一：Docker 一键部署（推荐）

### 1. 准备云服务器

**推荐配置：**

| 云平台 | 推荐产品 | 最低配置 | 月费参考 |
|--------|----------|----------|----------|
| 阿里云 | 轻量应用服务器 | 2核2G / 40G SSD | ~58元 |
| 腾讯云 | 轻量应用服务器 | 2核2G / 40G SSD | ~53元 |
| 华为云 | 云耀云服务器 | 2核2G / 40G SSD | ~60元 |

> ⚠️ 最低 2G 内存！sentence-transformers 模型加载需要 ~500MB。

**系统选择：** Ubuntu 22.04 LTS 或 CentOS 7.9+

### 2. 服务器初始化

```bash
# SSH 登录服务器
ssh root@你的服务器IP

# ===== Ubuntu =====
# 更新系统
apt update && apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com | bash

# 安装 Docker Compose
apt install docker-compose-plugin -y

# 启动 Docker
systemctl enable docker && systemctl start docker

# ===== 国内镜像加速（可选，避免 Docker Hub 拉取慢）=====
# 编辑 /etc/docker/daemon.json，添加：
# {
#   "registry-mirrors": [
#     "https://docker.1ms.run",
#     "https://docker.xuanyuan.me"
#   ]
# }
# systemctl daemon-reload && systemctl restart docker
```

### 3. 上传项目

```bash
# 在服务器上
mkdir -p /opt/apps && cd /opt/apps

# ===== 方式 A：从本地上传（在本地终端执行）=====
# scp -r E:/pycharm编程/AI_Agent/AI_Resume_Agent root@你的服务器IP:/opt/apps/

# ===== 方式 B：从 GitHub 克隆（如果已推送到 GitHub）=====
# git clone https://github.com/你的用户名/AI_Resume_Agent.git
# cd AI_Resume_Agent
```

### 4. 配置环境变量

```bash
cd /opt/apps/AI_Resume_Agent

# 创建 .env 文件（填你自己的 API Key）
cat > .env << 'EOF'
# LLM API 密钥
DEEPSEEK_API_KEY=sk-你的deepseek-key
OPENAI_API_KEY=sk-你的openai-key（可选）
LLM_PROVIDER=deepseek

# HuggingFace 镜像（国内用户必须）
HF_ENDPOINT=https://hf-mirror.com

# RAG 参数（保持默认即可）
TOP_K_RETRIEVAL=5
WEIGHT_PRESET=balanced
TEMPERATURE=0.1
EOF
```

### 5. 放置文档数据

```bash
# 把你的简历、项目文档等放到 data/raw/ 目录
# 支持 PDF / DOCX / MD / TXT 格式
ls data/raw/
# 示例输出: 01_简历_张三.pdf  02_项目_xxx.md  03_博客_xxx.md
```

### 6. 构建并启动

```bash
# 构建镜像（首次需要 5-10 分钟，会下载模型）
docker compose build

# 启动所有服务
docker compose up -d

# 查看启动日志
docker compose logs -f

# 看到以下输出表示成功：
# ✅ 索引文件已就绪
# ✅ 所有服务已启动
```

### 7. 测试访问

```bash
# 测试 API 健康检查
curl http://localhost/api/health

# 测试 API 问答
curl -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"请做自我介绍"}'
```

浏览器访问 `http://你的服务器IP` 即可看到面试助手界面！

### 8. 常用运维命令

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f --tail=50 api     # API 日志
docker compose logs -f --tail=50 app     # 前端日志

# 重启服务
docker compose restart

# 更新文档后重建索引
docker compose exec api python -m src.build_index --sync

# 停止服务
docker compose down

# 完全重建（清除向量库）
docker compose down -v
docker compose up -d --build
```

---

## 方案二：传统手动部署

适合不想用 Docker 的用户。

### 1. 服务器准备

```bash
# 安装 Python 3.10+
apt install python3.10 python3.10-venv python3-pip -y

# 安装 Nginx
apt install nginx -y
```

### 2. 部署后端 (FastAPI)

```bash
cd /opt/apps/AI_Resume_Agent

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖（使用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 构建索引
python -m src.build_index --sync

# 安装 systemd 服务
cat > /etc/systemd/system/resume-api.service << 'EOF'
[Unit]
Description=AI Resume Agent API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/apps/AI_Resume_Agent
EnvironmentFile=/opt/apps/AI_Resume_Agent/.env
ExecStart=/opt/apps/AI_Resume_Agent/venv/bin/python run_api.py --host 0.0.0.0 --port 8000 --no-reload
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable resume-api
systemctl start resume-api
```

### 3. 部署前端 (Streamlit)

```bash
# 安装 systemd 服务
cat > /etc/systemd/system/resume-app.service << 'EOF'
[Unit]
Description=AI Resume Agent Streamlit
After=network.target resume-api.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/apps/AI_Resume_Agent
Environment="API_BASE_URL=http://127.0.0.1:8000"
EnvironmentFile=/opt/apps/AI_Resume_Agent/.env
ExecStart=/opt/apps/AI_Resume_Agent/venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable resume-app
systemctl start resume-app
```

### 4. 配置 Nginx

```bash
cat > /etc/nginx/sites-available/resume << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    # API 后端
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }

    # Streamlit 前端
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 120s;
    }

    # WebSocket
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/resume /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

---

## 方案三：Streamlit Cloud（免费方案）

> ⚠️ 此方案前端免费，但后端仍需服务器。适合临时展示。

### 1. 后端部署

随便选一台最便宜的云服务器（1核1G 即可），只跑 FastAPI：

```bash
# 在服务器上
git clone https://github.com/你的用户名/AI_Resume_Agent.git
cd AI_Resume_Agent
pip install -r requirements.txt
python -m src.build_index --sync

# 用 nohup 后台运行（简单但不稳定）
nohup python run_api.py --host 0.0.0.0 --port 8000 --no-reload > api.log 2>&1 &

# 生产环境建议用 systemd（参考方案二）
```

### 2. 前端部署到 Streamlit Cloud

1. 把项目推送到 **GitHub 公开仓库**
2. 在 `.streamlit/secrets.toml` 中配置后端的公网地址：
   ```toml
   API_BASE_URL = "http://你的服务器IP:8000"
   ```
3. 打开 [share.streamlit.io](https://share.streamlit.io)
4. 关联 GitHub 仓库，选择 `app/streamlit_app.py` 作为入口
5. 部署后会得到一个 `xxx.streamlit.app` 域名

> ⚠️ Streamlit Cloud 免费版有休眠机制，15 分钟无访问会自动休眠。

---

## 域名 + HTTPS 配置

### 1. 购买域名

- 阿里云万网：https://wanwang.aliyun.com
- 腾讯云 DNSPod：https://dnspod.cloud.tencent.com
- 价格：`.com` ~60元/年，`.cn` ~30元/年

> ⚠️ **国内域名需要备案**（约 15-20 个工作日）。如果想快速上线，可以：
> - 先用 IP 访问
> - 或者购买海外域名（Namesilo、Namecheap 等，无需备案）

### 2. 域名解析

在域名管理后台添加 A 记录：

| 主机记录 | 记录类型 | 记录值 |
|----------|----------|--------|
| `@` | A | 你的服务器IP |
| `www` | A | 你的服务器IP |

### 3. 配置 HTTPS（Let's Encrypt 免费证书）

```bash
# 安装 certbot
apt install certbot python3-certbot-nginx -y

# 先修改 nginx 配置中的 server_name
# server_name your-domain.com;
# nginx -t && nginx -s reload

# 获取证书并自动配置
certbot --nginx -d your-domain.com

# 证书会自动续期，无需手动操作
```

### 4. 更新前端 API 地址（Docker 方式）

如果使用域名 + Nginx 反向代理，前端不需要直接访问后端 8000 端口，统一走 nginx 代理。编辑 `docker-compose.yml`：

```yaml
# 将 app 服务的 API_BASE_URL 改为空字符串（使用相对路径）
# 这样前端请求走 nginx 的 /api/ 路径
environment:
  - API_BASE_URL=
```

> 💡 `API_BASE_URL=` 为空时，chat.py 会使用 `"" + "/api/chat"` = `"/api/chat"`（相对路径），浏览器会自动拼接当前域名。

---

## 安全注意事项

### 🔒 必须做的

1. **`.env` 文件不要提交到 Git**
   - API Key 泄露 = 被人盗刷额度
   - `.gitignore` 已包含 `.env`

2. **修改默认端口/限制访问（可选）**
   ```bash
   # 云服务器安全组：只开放 80 和 443 端口
   # 不要开放 8000、8501 端口到公网！
   ```

3. **Streamlit 关闭文件上传等危险功能**
   - 当前项目未使用，无需额外配置

4. **设置 API 访问限流（可选）**
   - 防止恶意刷接口
   - 可在 Nginx 中配置 `limit_req_zone`

### 🛡️ 推荐的

5. **用 Nginx 限制单 IP 请求频率：**
   ```nginx
   # 在 nginx.conf 的 http 块中添加
   limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;

   # 在 /api/ location 块中添加
   # limit_req zone=api_limit burst=5 nodelay;
   ```

---

## 常见问题

### Q1: 服务器内存不够怎么办？
- 最低 2G 内存。1G 服务器可能 OOM
- 建议开启 swap：`fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`

### Q2: Docker 构建时下载模型太慢？
- `.env` 中已配置 `HF_ENDPOINT=https://hf-mirror.com`（HuggingFace 国内镜像）
- Dockerfile 中 pip 使用了清华镜像源

### Q3: 面试官访问慢怎么办？
- 选择 BGP 多线服务器（阿里云/腾讯云默认）
- 静态资源走 CDN（可选）

### Q4: 如何更新文档后刷新索引？
```bash
# Docker 方式
docker compose exec api python -m src.build_index --sync

# 手动部署方式
cd /opt/apps/AI_Resume_Agent
source venv/bin/activate
python -m src.build_index --sync
systemctl restart resume-api
```

### Q5: 想让面试官看到特定答案怎么办？
- 编辑 `prompts/system_prompt.toml`，修改系统提示词
- 编辑 `data/raw/` 目录下的文档，确保自己的简历信息完整准确
- 运行 `python -m src.evaluator --judge` 检查回答质量

### Q6: 服务器费用太贵了？
- 阿里云/腾讯云新用户首年优惠，轻量服务器 ~50-60元/月
- 使用按量计费，面试结束后销毁（几块钱一天）
- Streamlit Cloud 免费方案（前端免费，后端用小配置服务器）

---

## 🎯 部署后检查清单

- [ ] 浏览器访问 IP/域名能看到 Streamlit 界面
- [ ] 侧边栏能切换 LLM Provider
- [ ] 输入"自我介绍"能正常返回回答
- [ ] 回答下方能看到引用的文档来源
- [ ] API 健康检查正常（`/api/health` 返回 200）
- [ ] HTTPS 证书生效（有域名情况下）
- [ ] `.env` 文件不在公网可访问的路径

---

## 📞 需要帮助？

- 部署问题：检查 `docker compose logs` 查看错误日志
- RAG 检索问题：运行 `python -m src.evaluator --extended` 查看检索质量
- LLM 问题：检查 `.env` 中 API Key 是否正确，额度是否充足
