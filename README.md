# AUSLegalSearchv2 – End-to-End Secure Legal AI Deployment (Oracle Cloud/Ubuntu)

---

## Architecture Overview

```
                      [ Route53 (DNS) ]
                              |
                   auslegal.oraclecloudserver.com
                              |
           +------------------+-------------------+
           |                                     |
       [OCI Public Load Balancer]            (Alternative) [Nginx (public VM)]
           |                                     |
  (SSL Termination: Let's Encrypt)        (SSL Term: Let's Encrypt / Nginx)
           |                                     |
           +--------WAF (Web Application Firewall)+
           |
  (HTTP Proxy to backend set)
           |
     +--------------+
     | Ubuntu Server| (Private IP: e.g. 10.150.1.82)
     +--------------+
           |
   [Streamlit App :8501]
           |
   [PGvector/PostgreSQL DB]
```
**Flow Explanation:**
- DNS (Route53) points your domain/subdomain to the OCI Load Balancer.
- The LB handles all public HTTP/HTTPS, offloading SSL (with Let's Encrypt certs) and optionally applying a WAF policy for security.
- Requests are forwarded over HTTP to the backend server (private network).
- The backend runs the Python3/Streamlit app (virtualenv) and database.
- Users interact over secure HTTPS; Streamlit is never exposed directly.

---

## 1. Requirements & What You Get

- Secure, scalable legal search and RAG-powered summarization application.
- OAuth-ready, extensible user model, session-tracked chat.
- Leverages ollama/LLAMA for RAG inference.
- Modern Python, Streamlit, SQLAlchemy, pgvector, Certbot/Let's Encrypt, OCI/Route53.
- All steps are auditable, repeatable, and align with enterprise patterns.

---

## 2. Backend Server Setup (Ubuntu)

### a. Install System Packages & Whitelist Port
```sh
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git postgresql libpq-dev gcc unzip curl -y
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```
### b. Prepare Python Virtual Environment
```sh
git clone https://github.com/shadabshaukat/auslegalsearchv2.git
cd auslegalsearchv2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### c. Configure Environment (.env, .streamlit/config.toml)
- Edit `.streamlit/config.toml` for Streamlit binding (e.g., `address = "localhost"`).
- For `0.0.0.0` binding comment out all lines in the file `.streamlit/config.toml`
- Configure PostgreSQL (create DB, user as needed).
- Set DB environment variables
```
export AUSLEGALSEARCH_DB_HOST=localhost
export AUSLEGALSEARCH_DB_PORT=5432
export AUSLEGALSEARCH_DB_USER=postgres       
export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'  
export AUSLEGALSEARCH_DB_NAME=postgres
```

### d. Run and Persist the App (optional: with systemd)
```sh
streamlit run app.py
# Or create a systemd service to run the app on boot.
```
---

## 3. Database (PostgreSQL/pgvector)

- Set up PostgreSQL 15+ and install [pgvector](https://github.com/pgvector/pgvector).
- Configure user/password/roles and set AUSLEGALSEARCH_DB_URL env variable, or set in code.

---

## 4. SSL, DNS, and Load Balancer (OCI Approach, Recommended)

### a. **Route53:**
- Define your domain/subdomain and set A-record to your OCI load balancer's public IP.

### b. **OCI Load Balancer:**
- Configure TCP/HTTP listener on port 80, and add backend set with your app's internal (private) IP and port 8501.
- Add **WAF policy** (recommended): restrict by IP/CIDR, SQLi/XSS protection, bot block, etc.
- [Optional] Allow HTTP→HTTPS redirect rule.

### c. **Obtain Let's Encrypt SSL Cert (with DNS Challenge)**
- On any Linux box (not necessarily your backend) with AWS Route53 credentials, run:

```sh
sudo apt install certbot python3-certbot-dns-route53
aws configure # with Route53 permissions
sudo certbot certonly --dns-route53 -d auslegal.oraclecloudserver.com
```
- See certbot_route53_aws_credentials.md for IAM policy/example.

### d. **Upload Certs to OCI Load Balancer**
- Use OCI Console or CLI, upload fullchain.pem and privkey.pem as certificate, attach to HTTPS:443 listener.
- Automate renewal with cron/systemd and OCI CLI script (see setup_letsencrypt_oracle_lb.md).

---

## 5. (Alternative) Nginx Reverse Proxy + Let's Encrypt (Simple/Dev)

- On the public VM, install nginx and certbot.
- See setup_letsencrypt_streamlit_nginx.md for complete reverse proxy config.

---

## 6. WAF Policy (Good Practice)

- In OCI LB management, create/attach a WAF policy to your LB with:
  - IP restrictions, geoblocking
  - SQLi/XSS/CSRF protections
  - Rate limiting, bot defense
- [OCI WAF Docs](https://docs.oracle.com/en-us/iaas/Content/WAF/Tasks/wafconfigure.htm)

---

## 7. Application Logic and RAG Pipeline

### a. **RAG (Retrieval Augmented Generation)**
- Documents are ingested, chunked, embedded via model (MiniLM or other).
- Embeddings stored in pgvector for efficient hybrid search (BM25+vector).
- User queries are matched for both semantic similarity and keyword relevance.
- Top chunks are retrieved, composed as context for the LLM (LLAMA3/4 via Ollama).
- RAG prompt includes retrieved evidence, with enforced grounding/citations.

### b. **Chat & Session Management**
- Each chat persists messages and context to PostgreSQL (username, question, chat history, LLM params).
- Supports session resumption, full conversation memory, secure user login.

### c. **App Structure**
- `app.py`: Document search, ingestion control.
- `pages/chat.py`: Chatbot/RAG experience.
- `db/store.py`: ORM + vector search layer.
- `embedding/`, `rag/`, etc: Pluggable vector and retrieval pipelines.

**Flow:**
1. User logs in or is authenticated.
2. User can ingest new documents into corpus (legal PDFs, HTML, etc.).
3. User asks a question; top-matching docs are retrieved, shown as "evidence".
4. RAG LLM answers, strictly grounded in the retrieved data, with citations.
5. All conversation/usage is session-tracked in DB (audit, learning, future fine-tuning).

---

## 8. Business Use Cases

- Legal research automation: fast, explainable case law and legislative guidance.
- Litigation preparation: retrieve, ground, and summarize precedent and statute.
- Compliance: trace document sources/evidence with AI-augmented summaries.
- Internal knowledge management: rapid search for distributed legal and compliance teams.
- Client question intake: safe, auditable, lawyer-in-the-loop summaries.

---

## 9. Extensibility & Hardening

- Modular: add new ingestion pipelines, LLM connectors, embedding models easily.
- Secure by design: hidden backend, strong DB roles, user auth, WAF.
- Enables easy deployment to Oracle, AWS, or other private/public cloud platforms.
- Full logging, error-handling, and monitoring hooks included for ops teams.

---

## 10. Troubleshooting/Recovery

- Logs: streamlit/debug, letsencrypt, oci-cli, and DB/postgres logs.
- Certificates: check `/etc/letsencrypt/live/`, use `certbot renew --dry-run`.
- OCI: check LB backend health, cert attach status, WAF events.
- Streamlit: isolate with `venv`, restart app and reload Nginx/LB as needed.

---

## 11. Example Systemd Service (If Needed)

Create `/etc/systemd/system/auslegalsearch.service`:
```
[Unit]
Description=AUSLegalSearchv2 Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/auslegalsearchv2
ExecStart=/home/ubuntu/auslegalsearchv2/venv/bin/streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable with:
```sh
sudo systemctl daemon-reload
sudo systemctl enable --now auslegalsearch
```
---

## 12. Contact and Contribution

- For support/issues, create a GitHub issue or contact the maintainer.
- PRs for new retrievers, UI improvements, or legal features are welcome.

---

**Deploy with confidence – this architecture is secure, auditable, and enterprise-ready.**
