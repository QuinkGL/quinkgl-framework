#!/bin/bash

# QuinkGL Docker Test Script
# Bu script Docker ile çoklu peer testi için hızlı başlangıç sağlar

set -e

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   QuinkGL Docker Test Environment    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════╝${NC}"
echo ""

# Docker kontrolü
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker bulunamadı!${NC}"
    echo -e "${YELLOW}Lütfen Docker Desktop'ı kurun: https://www.docker.com/products/docker-desktop/${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose bulunamadı!${NC}"
    echo -e "${YELLOW}Lütfen Docker Desktop'ı kurun: https://www.docker.com/products/docker-desktop/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker kurulu${NC}"
echo ""

# Menü
echo "Ne yapmak istersiniz?"
echo ""
echo "1) Docker image'ını oluştur (ilk kez çalıştırma)"
echo "2) 10 peer başlat (arka planda)"
echo "3) 10 peer başlat (logları göster)"
echo "4) Belirli sayıda peer başlat"
echo "5) Peer loglarını göster"
echo "6) Bir peer'a bağlan (interaktif)"
echo "7) Peer'ları durdur"
echo "8) Tüm container'ları temizle"
echo "9) Docker sistem bilgisi"
echo "0) Çıkış"
echo ""
read -p "Seçiminiz (0-9): " choice

case $choice in
    1)
        echo -e "${BLUE}🔨 Docker image'ı oluşturuluyor...${NC}"
        docker compose build
        echo -e "${GREEN}✅ Image başarıyla oluşturuldu!${NC}"
        echo -e "${YELLOW}Şimdi '2' veya '3' seçeneği ile peer'ları başlatabilirsiniz.${NC}"
        ;;
    
    2)
        echo -e "${BLUE}🚀 10 peer arka planda başlatılıyor...${NC}"
        docker compose up -d
        echo ""
        echo -e "${GREEN}✅ Peer'lar başlatıldı!${NC}"
        echo ""
        echo "Logları görmek için: docker compose logs -f"
        echo "Bir peer'a bağlanmak için: docker attach quinkgl-peer1"
        echo "Durdurmak için: docker compose down"
        ;;
    
    3)
        echo -e "${BLUE}🚀 10 peer başlatılıyor (loglar gösteriliyor)...${NC}"
        echo -e "${YELLOW}Durdurmak için: Ctrl+C${NC}"
        echo ""
        docker compose up
        ;;
    
    4)
        read -p "Kaç peer başlatmak istersiniz? (1-10): " num_peers
        if [ "$num_peers" -lt 1 ] || [ "$num_peers" -gt 10 ]; then
            echo -e "${RED}❌ Geçersiz sayı! 1-10 arası olmalı.${NC}"
            exit 1
        fi
        
        peers=""
        for i in $(seq 1 $num_peers); do
            peers="$peers peer$i"
        done
        
        echo -e "${BLUE}🚀 $num_peers peer başlatılıyor...${NC}"
        docker compose up -d $peers
        echo -e "${GREEN}✅ $num_peers peer başlatıldı!${NC}"
        ;;
    
    5)
        echo -e "${BLUE}📋 Peer logları gösteriliyor...${NC}"
        echo -e "${YELLOW}Durdurmak için: Ctrl+C${NC}"
        echo ""
        docker compose logs -f
        ;;
    
    6)
        echo "Hangi peer'a bağlanmak istersiniz?"
        docker compose ps
        echo ""
        read -p "Peer numarası (1-10): " peer_num
        
        if [ "$peer_num" -lt 1 ] || [ "$peer_num" -gt 10 ]; then
            echo -e "${RED}❌ Geçersiz peer numarası!${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}🔗 peer$peer_num'e bağlanılıyor...${NC}"
        echo -e "${YELLOW}Çıkmak için: Ctrl+P ardından Ctrl+Q${NC}"
        echo ""
        sleep 2
        docker attach quinkgl-peer$peer_num
        ;;
    
    7)
        echo -e "${BLUE}🛑 Peer'lar durduruluyor...${NC}"
        docker compose down
        echo -e "${GREEN}✅ Tüm peer'lar durduruldu!${NC}"
        ;;
    
    8)
        echo -e "${RED}⚠️  Tüm container'lar ve image'lar silinecek!${NC}"
        read -p "Emin misiniz? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo -e "${BLUE}🗑️  Temizleniyor...${NC}"
            docker compose down -v
            docker system prune -a -f
            echo -e "${GREEN}✅ Temizlik tamamlandı!${NC}"
        else
            echo -e "${YELLOW}İptal edildi.${NC}"
        fi
        ;;
    
    9)
        echo -e "${BLUE}📊 Docker Sistem Bilgisi${NC}"
        echo ""
        echo "=== Çalışan Container'lar ==="
        docker compose ps
        echo ""
        echo "=== Resource Kullanımı ==="
        docker stats --no-stream
        echo ""
        echo "=== Disk Kullanımı ==="
        docker system df
        ;;
    
    0)
        echo -e "${GREEN}👋 Görüşmek üzere!${NC}"
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Geçersiz seçim!${NC}"
        exit 1
        ;;
esac
