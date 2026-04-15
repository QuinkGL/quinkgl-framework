@echo off
REM QuinkGL Docker Test Script for Windows
REM Bu script Docker ile çoklu peer testi için hızlı başlangıç sağlar

echo ╔═══════════════════════════════════════╗
echo ║   QuinkGL Docker Test Environment    ║
echo ╚═══════════════════════════════════════╝
echo.

REM Docker kontrolü
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker bulunamadı!
    echo Lütfen Docker Desktop'ı kurun: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

docker compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose bulunamadı!
    echo Lütfen Docker Desktop'ı kurun: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo ✅ Docker kurulu
echo.

REM Menü
echo Ne yapmak istersiniz?
echo.
echo 1) Docker image'ını oluştur (ilk kez çalıştırma)
echo 2) 10 peer başlat (arka planda)
echo 3) 10 peer başlat (logları göster)
echo 4) Belirli sayıda peer başlat
echo 5) Peer loglarını göster
echo 6) Bir peer'a bağlan (interaktif)
echo 7) Peer'ları durdur
echo 8) Tüm container'ları temizle
echo 9) Docker sistem bilgisi
echo 0) Çıkış
echo.

set /p choice="Seçiminiz (0-9): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto start_bg
if "%choice%"=="3" goto start_fg
if "%choice%"=="4" goto start_custom
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto attach
if "%choice%"=="7" goto stop
if "%choice%"=="8" goto clean
if "%choice%"=="9" goto info
if "%choice%"=="0" goto exit
goto invalid

:build
echo 🔨 Docker image'ı oluşturuluyor...
docker compose build
echo ✅ Image başarıyla oluşturuldu!
echo Şimdi '2' veya '3' seçeneği ile peer'ları başlatabilirsiniz.
pause
exit /b 0

:start_bg
echo 🚀 10 peer arka planda başlatılıyor...
docker compose up -d
echo.
echo ✅ Peer'lar başlatıldı!
echo.
echo Logları görmek için: docker compose logs -f
echo Bir peer'a bağlanmak için: docker attach quinkgl-peer1
echo Durdurmak için: docker compose down
pause
exit /b 0

:start_fg
echo 🚀 10 peer başlatılıyor (loglar gösteriliyor)...
echo Durdurmak için: Ctrl+C
echo.
docker compose up
exit /b 0

:start_custom
set /p num_peers="Kaç peer başlatmak istersiniz? (1-10): "
if %num_peers% lss 1 goto invalid_num
if %num_peers% gtr 10 goto invalid_num

set peers=
for /l %%i in (1,1,%num_peers%) do (
    call set peers=%%peers%% peer%%i
)

echo 🚀 %num_peers% peer başlatılıyor...
docker compose up -d %peers%
echo ✅ %num_peers% peer başlatıldı!
pause
exit /b 0

:invalid_num
echo ❌ Geçersiz sayı! 1-10 arası olmalı.
pause
exit /b 1

:logs
echo 📋 Peer logları gösteriliyor...
echo Durdurmak için: Ctrl+C
echo.
docker compose logs -f
exit /b 0

:attach
echo Hangi peer'a bağlanmak istersiniz?
docker compose ps
echo.
set /p peer_num="Peer numarası (1-10): "

if %peer_num% lss 1 goto invalid_peer
if %peer_num% gtr 10 goto invalid_peer

echo 🔗 peer%peer_num%'e bağlanılıyor...
echo Çıkmak için: Ctrl+P ardından Ctrl+Q
echo.
timeout /t 2 >nul
docker attach quinkgl-peer%peer_num%
exit /b 0

:invalid_peer
echo ❌ Geçersiz peer numarası!
pause
exit /b 1

:stop
echo 🛑 Peer'lar durduruluyor...
docker compose down
echo ✅ Tüm peer'lar durduruldu!
pause
exit /b 0

:clean
echo ⚠️  Tüm container'lar ve image'lar silinecek!
set /p confirm="Emin misiniz? (y/N): "
if /i "%confirm%"=="y" goto do_clean
echo İptal edildi.
pause
exit /b 0

:do_clean
echo 🗑️  Temizleniyor...
docker compose down -v
docker system prune -a -f
echo ✅ Temizlik tamamlandı!
pause
exit /b 0

:info
echo 📊 Docker Sistem Bilgisi
echo.
echo === Çalışan Container'lar ===
docker compose ps
echo.
echo === Resource Kullanımı ===
docker stats --no-stream
echo.
echo === Disk Kullanımı ===
docker system df
pause
exit /b 0

:invalid
echo ❌ Geçersiz seçim!
pause
exit /b 1

:exit
echo 👋 Görüşmek üzere!
exit /b 0
