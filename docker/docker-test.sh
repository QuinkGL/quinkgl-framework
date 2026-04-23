#!/bin/bash

# QuinkGL Docker Test Script
# This script provides a quick start for multi-peer testing with Docker

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   QuinkGL Docker Test Environment    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════╝${NC}"
echo ""

# Docker check
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    echo -e "${YELLOW}Please install Docker Desktop: https://www.docker.com/products/docker-desktop/${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found!${NC}"
    echo -e "${YELLOW}Please install Docker Desktop: https://www.docker.com/products/docker-desktop/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker installed${NC}"
echo ""

# Menu
echo "What would you like to do?"
echo ""
echo "1) Build Docker image (first time run)"
echo "2) Start 10 peers (background)"
echo "3) Start 10 peers (show logs)"
echo "4) Start specific number of peers"
echo "5) Show peer logs"
echo "6) Connect to a peer (interactive)"
echo "7) Stop peers"
echo "8) Clean all containers"
echo "9) Docker system info"
echo "0) Exit"
echo ""
read -p "Your choice (0-9): " choice

case $choice in
    1)
        echo -e "${BLUE}🔨 Building Docker image...${NC}"
        docker compose build
        echo -e "${GREEN}✅ Image built successfully!${NC}"
        echo -e "${YELLOW}Now you can start peers with option '2' or '3'.${NC}"
        ;;

    2)
        echo -e "${BLUE}🚀 Starting 10 peers in background...${NC}"
        docker compose up -d
        echo ""
        echo -e "${GREEN}✅ Peers started!${NC}"
        echo ""
        echo "To view logs: docker compose logs -f"
        echo "To connect to a peer: docker attach quinkgl-peer1"
        echo "To stop: docker compose down"
        ;;

    3)
        echo -e "${BLUE}🚀 Starting 10 peers (showing logs)...${NC}"
        echo -e "${YELLOW}To stop: Ctrl+C${NC}"
        echo ""
        docker compose up
        ;;

    4)
        read -p "How many peers do you want to start? (1-10): " num_peers
        if [ "$num_peers" -lt 1 ] || [ "$num_peers" -gt 10 ]; then
            echo -e "${RED}❌ Invalid number! Must be between 1-10.${NC}"
            exit 1
        fi

        peers=""
        for i in $(seq 1 $num_peers); do
            peers="$peers peer$i"
        done

        echo -e "${BLUE}🚀 Starting $num_peers peers...${NC}"
        docker compose up -d $peers
        echo -e "${GREEN}✅ $num_peers peers started!${NC}"
        ;;

    5)
        echo -e "${BLUE}📋 Showing peer logs...${NC}"
        echo -e "${YELLOW}To stop: Ctrl+C${NC}"
        echo ""
        docker compose logs -f
        ;;

    6)
        echo "Which peer do you want to connect to?"
        docker compose ps
        echo ""
        read -p "Peer number (1-10): " peer_num

        if [ "$peer_num" -lt 1 ] || [ "$peer_num" -gt 10 ]; then
            echo -e "${RED}❌ Invalid peer number!${NC}"
            exit 1
        fi

        echo -e "${BLUE}🔗 Connecting to peer$peer_num...${NC}"
        echo -e "${YELLOW}To exit: Ctrl+P then Ctrl+Q${NC}"
        echo ""
        sleep 2
        docker attach quinkgl-peer$peer_num
        ;;

    7)
        echo -e "${BLUE}🛑 Stopping peers...${NC}"
        docker compose down
        echo -e "${GREEN}✅ All peers stopped!${NC}"
        ;;

    8)
        echo -e "${RED}⚠️  All containers and images will be deleted!${NC}"
        read -p "Are you sure? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo -e "${BLUE}🗑️  Cleaning...${NC}"
            docker compose down -v
            docker system prune -a -f
            echo -e "${GREEN}✅ Cleanup complete!${NC}"
        else
            echo -e "${YELLOW}Cancelled.${NC}"
        fi
        ;;

    9)
        echo -e "${BLUE}📊 Docker System Info${NC}"
        echo ""
        echo "=== Running Containers ==="
        docker compose ps
        echo ""
        echo "=== Resource Usage ==="
        docker stats --no-stream
        echo ""
        echo "=== Disk Usage ==="
        docker system df
        ;;

    0)
        echo -e "${GREEN}👋 Goodbye!${NC}"
        exit 0
        ;;

    *)
        echo -e "${RED}❌ Invalid selection!${NC}"
        exit 1
        ;;
esac
