#!/bin/bash
set -euo pipefail

echo "=== RAG System - Oracle Cloud Deployment Setup ==="
echo ""

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group changes."
fi

echo "Docker version: $(docker --version)"
echo ""

APP_DIR="${APP_DIR:-/opt/rag-system}"

if [ ! -d "$APP_DIR" ]; then
    echo "Cloning repository to $APP_DIR..."
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER:$USER" "$APP_DIR"
    git clone https://github.com/aditya-kumaran/rag-system.git "$APP_DIR"
fi

cd "$APP_DIR"

if [ ! -f "$APP_DIR/.env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "IMPORTANT: Edit $APP_DIR/.env and add your GROQ_API_KEY"
    echo "  nano $APP_DIR/.env"
    echo ""
fi

if [ ! -d "$APP_DIR/data/chroma_db" ]; then
    echo ""
    echo "WARNING: No data directory found at $APP_DIR/data/"
    echo "You need to copy your local data directory to the server:"
    echo "  scp -r ./data/ user@<server-ip>:$APP_DIR/data/"
    echo ""
    echo "The data directory should contain:"
    echo "  - chroma_db/        (ChromaDB vector index)"
    echo "  - graph/            (citation graph)"
    echo "  - papers_metadata.json"
    echo "  - bm25_index.pkl"
    echo "  - sessions.db       (optional, will be created)"
    echo ""
fi

echo "Opening firewall ports..."
sudo iptables -I INPUT -p tcp --dport 7860 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT 2>/dev/null || true

if command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-port=7860/tcp 2>/dev/null || true
    sudo firewall-cmd --permanent --add-port=8000/tcp 2>/dev/null || true
    sudo firewall-cmd --reload 2>/dev/null || true
fi

echo ""
echo "IMPORTANT: You must also open ports 7860 and 8000 in your Oracle Cloud"
echo "Security List (VCN > Subnet > Security List > Add Ingress Rules):"
echo "  - Source CIDR: 0.0.0.0/0, Protocol: TCP, Dest Port: 7860"
echo "  - Source CIDR: 0.0.0.0/0, Protocol: TCP, Dest Port: 8000"
echo ""

echo "Building and starting containers..."
docker compose up -d --build

echo ""
echo "=== Deployment Complete ==="
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-server-ip>")
echo "UI:  http://$PUBLIC_IP:7860"
echo "API: http://$PUBLIC_IP:8000"
echo "API docs: http://$PUBLIC_IP:8000/docs"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f        # View logs"
echo "  docker compose restart        # Restart services"
echo "  docker compose down           # Stop services"
echo "  docker compose up -d --build  # Rebuild and restart"
