# QuinkGL Başlangıç Rehberi

Bu rehber, QuinkGL ile ilk merkeziyetsiz öğrenme (decentralized learning) swarm'ınızı kurmak isteyen bir geliştirici/operatör için yazılmıştır. Manifest nedir, creator key ne işe yarar, model hash'i nasıl alınır, script zorunlu mu, veri nasıl bağlanır gibi soruların tümüne adım adım cevap verir.

---

## 1. Temel Kavramlar

### 1.1 Swarm Nedir?

**Swarm**, aynı manifest'i (eğitim protokolünü) paylaşan ve birbirleriyle doğrudan P2P iletişim kurarak model eğiten peer'lerin oluşturduğu gruptur. Merkezi bir sunucu yoktur; her peer diğer peer'lere modelinin ağırlıklarını gönderir ve alır.

### 1.2 Manifest Nedir?

**Manifest** (`.qgl` dosyası), swarm'ın "anayasasıdır." İçinde şunlar yazar:

- Görev tipi (sınıflandırma, regresyon, segmentasyon, deteksiyon)
- Model mimarisi hash'i (hangi mimariye izin var)
- Aggregation stratejisi (FedAvg, EntropyWeightedAvg, Krum, vb.)
- Topology stratejisi (RandomTopology, AffinityTopology, CyclonTopology)
- Veri politikası (fingerprint, privacy level, collaboration mode)
- Görev şekli (input/output shape, label type)
- Creator imzası (manifest'i oluşturanın kim olduğunun kanıtı)

> **Önemli:** Manifest'in kendisi bir swarm değildir. Manifest, swarm'a katılma kurallarını tanımlayan bir "blueprint"tir. Swarm, manifest'i kullanan çalışan peer'lerin kendisidir.

### 1.3 Creator Key Nedir ve Ne İşe Yarar?

**Creator key**, manifest'i imzalayan kişinin/organizasyonun **Ed25519 private key**'idir.

**Ne işe yarar?**

1. **Kimlik doğrulama:** `creator_pubkey` alanı manifest'e yazılır. Peer'lar, manifest'in bu key tarafından imzalandığını doğrular.
2. **Trust On First Use (TOFU):** `--trust-policy tofu` kullanıldığında, peer ilk kez bir manifest gördüğünde creator pubkey'ini yerel önbelleğe alır. Eğer sonradan aynı manifest ismi farklı bir creator key ile gelirse, peer reddeder.
3. **Pinning:** `--trust-policy pinned` kullanıldığında, peer **sadece** belirtilen `--trusted-pubkey` ile imzalanmış manifest'leri kabul eder.

**Ne zaman kullanmalısınız?**

- Üretim ortamında (production) **mutlaka** kullanın. Aksi halde herhangi biri sahte bir manifest oluşturup swarm'ınıza sızabilir.
- Lokal testlerde (`--trust-policy open` veya `tofu`) keygen yapmadan da test edebilirsiniz, ancak manifest imzasız kalır.

### 1.4 Script (peer_script.py) Zorunlu mu?

**Evet, Mode B'de zorunludur.** QuinkGL'in çalışma modları vardır:

- **Mode A:** `--data` ile standart bir model ve veri seti kullanırsınız. (Şu an sınırlı destek)
- **Mode B:** `--script` ile kendi modelinizi, veri yükleyicinizi ve optimizer'ınızı tanımlarsınız. **Gerçek dünya kullanımı budur.**

`peer_script.py`'de bulunması zorunlu fonksiyonlar:

```python
def build_model(manifest, **kwargs):
    ...

def build_loaders(manifest, **kwargs):
    ...  # (train_loader, val_loader) tuple döner
```

Opsiyonel fonksiyonlar:

```python
def build_optimizer(manifest, model):
    ...

def on_round_end(round_idx, metrics):
    ...

def on_peer_discovered(peer_id):
    ...

def on_aggregation_done(peer_ids, sample_count):
    ...
```

---

## 2. Baştan Sona Tam Bir Örnek

Aşağıdaki örnekte, 5 peer'lık yerel bir test swarm'ı kuracağız. Her peer kendi cihazında eğitim yapacak ve modelleri birbirleriyle paylaşacak. **Gerçek torchvision veri setlerini** (MNIST ve CIFAR-10) kullanacağız, mock veri yerine.

### 2.1 Creator Key Oluşturma

```bash
quinkgl keygen --output creator.key
```

Çıktı:
```
Private key written with 0600 permissions. Treat this file as a secret...
ed25519:bfc5819e0264e22be8f1363794aa152a468123f7a797fa57decd88bdd21c0518
```

> **Güvenlik:** `creator.key` dosyasını asla git repoya eklemeyin. `.gitignore`'a ekleyin.

### 2.2 Model Mimarisi Hash'ini Alma

Model mimarisi hash'i, kullanacağınız mimarinin **SHA-256** değeridir. Bu, swarm'a "sadece bu mimariye sahip model kabul edilir" demenizi sağlar.

**Neye göre hash alınır?**

- **Model mimarisi:** Katman tipleri, sıraları, boyutları (ağırlıklar değil, sadece yapı).

**Nasıl alınır:**

QuinkGL'in gömülü helper'ını kullanabilirsiniz:

```python
from quinkgl.manifest import compute_arch_hash
import torch.nn as nn

class MyModel(nn.Module):
    ...

model = MyModel()
arch_hash = compute_arch_hash(model)
print(arch_hash)  # sha256:40f4a106...
```

Ya da CLI'dan doğrudan hesaplayabilirsiniz:

```bash
python -c "
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel
print(compute_arch_hash(MyModel()))
"
```

> **Önemli:** Eğer model mimarinizi değiştirirseniz (örn. yeni katman ekler), hash değişir ve eski manifest artık geçersiz olur. Yeni manifest oluşturmanız gerekir.

### 2.3 Manifest Oluşturma

```bash
quinkgl manifest create \
  --name demo-5peer \
  --task-type class \
  --input-shape 1,28,28 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:40f4a106862aa557fdbeb62a0daaa87f2b031acf93a2f9d2028e481c9607b3a5 \
  --aggregation EntropyWeightedAvg \
  --topology AffinityTopology \
  --sign-with creator.key \
  --output demo.qgl
```

**Parametrelerin anlamı:**

| Parametre | Açıklama |
|-----------|----------|
| `--name` | Swarm'ın adı |
| `--task-type` | Görev: `class` (sınıflandırma), `regr`, `seg`, `det` |
| `--input-shape` | Model girdisi: kanal, yükseklik, genişlik |
| `--output-shape` | Model çıktısı: sınıf sayısı |
| `--label-type` | Etiket tipi: `integer`, `float`, `one_hot` |
| `--model-framework` | `pytorch`, `tensorflow`, `custom` |
| `--model-arch-hash` | Model mimarisinin SHA-256 hash'i |
| `--aggregation` | Aggregation stratejisi |
| `--topology` | Peer seçim stratejisi |
| `--sign-with` | Creator private key (PEM dosyası) |
| `--output` | Çıktı manifest dosyası |

**Opsiyonel parametreler:**

```bash
  --round-limit 100              # Max round sayısı
  --byzantine-f 1                # Byzantine peer sayısı (tolerans)
  --expires-at 2025-12-31        # Manifest geçerlilik süresi
  --bootstrap-peer 192.168.1.5:7001  # Başlangıç peer'leri
```

### 2.4 Manifest'i Doğrulama

```bash
quinkgl manifest verify demo.qgl --trusted-pubkey ed25519:bfc5819e...
```

Çıktı:
```
Manifest valid.
Swarm ID: sha256:3a8f2e...
Signature: valid
```

### 2.5 Peer Script'i Yazma

#### Örnek A: MNIST (28×28 grayscale, 10 sınıf)

`mnist_peer_script.py`:

```python
"""Peer script for MNIST classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(manifest, **kwargs):
    return MNISTNet()


def build_loaders(manifest, **kwargs):
    batch_size = int(kwargs.get("batch_size", 32))
    data_root = kwargs.get("data_root", "./data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    val_ds = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_optimizer(manifest, model):
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def on_round_end(round_idx, metrics):
    loss = metrics.get("loss")
    acc = metrics.get("val_accuracy") or metrics.get("accuracy")
    tag = f"round={round_idx:03d}"
    if loss is not None:
        tag += f" loss={loss:.4f}"
    if acc is not None:
        tag += f" acc={acc:.3f}"
    print(tag, flush=True)


def on_peer_discovered(peer_id):
    print(f"[peer-discovered] {peer_id}", flush=True)


def on_aggregation_done(peer_ids, sample_count):
    print(f"[aggregated] peers={list(peer_ids)} samples={sample_count}", flush=True)
```

#### Örnek B: CIFAR-10 (32×32 RGB, 10 sınıf)

`cifar10_peer_script.py`:

```python
"""Peer script for CIFAR-10 classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(manifest, **kwargs):
    return CIFAR10Net()


def build_loaders(manifest, **kwargs):
    batch_size = int(kwargs.get("batch_size", 32))
    data_root = kwargs.get("data_root", "./data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    val_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_optimizer(manifest, model):
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def on_round_end(round_idx, metrics):
    loss = metrics.get("loss")
    acc = metrics.get("val_accuracy") or metrics.get("accuracy")
    tag = f"round={round_idx:03d}"
    if loss is not None:
        tag += f" loss={loss:.4f}"
    if acc is not None:
        tag += f" acc={acc:.3f}"
    print(tag, flush=True)


def on_peer_discovered(peer_id):
    print(f"[peer-discovered] {peer_id}", flush=True)


def on_aggregation_done(peer_ids, sample_count):
    print(f"[aggregated] peers={list(peer_ids)} samples={sample_count}", flush=True)
```

### 2.6 Tek Peer Başlatma

Bir terminal açın ve tek bir peer'i doğrudan çalıştırın:

```bash
quinkgl run \
  --manifest demo.qgl \
  --script mnist_peer_script.py \
  --node-id peer-1 \
  --port 7001 \
  --rounds 15 \
  --gossip-interval 12.0 \
  --trust-policy tofu \
  --script-arg data_root=./data \
  --checkpoint-dir ./ckpt/peer-1
```

**Bu komut ne yapar:**
- `demo.qgl` manifest'ini yükler
- Model ve veriyi `mnist_peer_script.py`'den yükler
- IPv8'i UDP port 7001'de başlatır
- 15 gossip learning round'u çalıştırır
- Checkpoint'leri `./ckpt/peer-1`'e kaydeder
- Round metriklerini stdout'a basar

**İlk çalıştırma notu:** `torchvision`, ilk başlatmada MNIST/CIFAR-10'i otomatik olarak `./data` dizinine indirir.

**İkinci peer çalıştırmak için** (ikinci bir terminal açın):

```bash
quinkgl run \
  --manifest demo.qgl \
  --script mnist_peer_script.py \
  --node-id peer-2 \
  --port 7002 \
  --rounds 15 \
  --gossip-interval 12.0 \
  --trust-policy tofu \
  --script-arg data_root=./data \
  --checkpoint-dir ./ckpt/peer-2
```

### 2.7 Durumu İzleme

```bash
# Aggregation logları
grep 'aggregated models' logs/peer-1.log

# Round başına accuracy
grep 'round=' logs/peer-1.log

# Keşfedilen peer'ler
grep 'peer-discovered' logs/peer-1.log

# Tüm peer'leri durdur
pkill -f 'quinkgl run'
```

---

## 3. Hash'ler Derinlemesine

### 3.1 Model Mimarisi Hash'i (`model_arch_hash`)

**Neye göre alınır?**

Modelin **yapısının** hash'i. Ağırlıklar (weights) hash'e dahil değildir; sadece:
- Katman tipleri ve sıraları
- Boyutlar (input/output feature count)
- Aktivasyon fonksiyonları

**Neden önemli?**

Swarm'daki tüm peer'lar aynı mimariyi kullanmalıdır. Aksi halde aggregation (ağırlık ortalaması) anlamsız olur.

**Nasıl alınır?**

QuinkGL'in gömülü helper'ını kullanın:

```python
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel

model = MyModel()
arch_hash = compute_arch_hash(model)
print(arch_hash)  # sha256:...
```

### 3.2 Veri Şeması Hash'i (`data_schema_hash`)

**Neye göre alınır?**

Veri setinin şeklinin hash'i. Girdi boyutu, kanal sayısı, etiket tipi gibi meta-veriler.

**Neden önemli?**

Peer'ler, veri şeması uyuşmayan diğer peer'leri otomatik olarak reddeder (güvenlik + uyumluluk).

**Nasıl alınır?**

QuinkGL otomatik üretir. Elle vermek isterseniz:

```python
from quinkgl.models import PyTorchModel

model_wrapper = PyTorchModel(MyModel())
schema_hash = model_wrapper.get_data_schema_hash()
print(schema_hash)  # sha256:0000... şeklinde
```

---

## 4. Trust Policy'ler

| Policy | Davranış | Kullanım senaryosu |
|--------|----------|-------------------|
| `open` | Her manifest'i kabul eder, imza kontrolu yapmaz | Hızlı testler |
| `tofu` | İlk gördüğü creator key'i önbelleğe alır, sonradan değişirse reddeder | Üretim (önerilen) |
| `pinned` | Sadece `--trusted-pubkey` ile verilen key'leri kabul eder | Yüksek güvenlik |

**TOFU örneği:**

```bash
quinkgl run --manifest demo.qgl --trust-policy tofu ...
```

**Pinned örneği:**

```bash
quinkgl run --manifest demo.qgl \
  --trust-policy pinned \
  --trusted-pubkey ed25519:bfc5819e0264... \
  ...
```

---

## 5. Sık Sorulan Sorular

**S: Manifest'i değiştirdim, eski peer'lar yeni manifest'i kabul eder mi?**

C: Manifest hash'i (swarm ID) değiştiği için eski peer'lar yeni manifest'i farklı bir swarm olarak görür. Peer'ları yeni manifest ile yeniden başlatmanız gerekir.

**S: Aynı creator key ile birden fazla manifest oluşturabilir miyim?**

C: Evet. Her farklı eğitim görevi için ayrı manifest oluşturabilirsiniz.

**S: Model mimarimi değiştirdim ama manifest'i güncellemek istemiyorum.**

C: `--strict-manifest false` ile çalıştırabilirsiniz ancak bu önerilmez. Aggregation hataları veya güvenlik sorunları yaşayabilirsiniz.

**S: Veri setim çok büyük, her peer'a kopyalamak zorunda mıyım?**

C: Hayır. Her peer kendi verisine sahip olur (federated learning'ın özü budur). `build_loaders` fonksiyonunda her peer kendi veri yolunu kullanır.

**S: `on_aggregation_done` ne zaman çağrılır?**

C: Bir peer, diğer peer'lerden model alıp aggregation tamamladığında çağrılır. `[aggregated] peers=[...]` logunu görürseniz, aggregation başarılıdır.

**S: IPv8 portunu 0 versem ne olur?**

C: İşletim sistemi rastgele boş bir port atar. Üretimde sabit port kullanın; discovery için gerekir.

---

## 6. Hızlı Referans Kartı

```bash
# Key oluştur
quinkgl keygen --output creator.key

# Model mimarisi hash'i hesapla
python -c "
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel
print(compute_arch_hash(MyModel()))
"

# Manifest oluştur
quinkgl manifest create \
  --name <name> --task-type class \
  --input-shape <C,H,W> --output-shape <classes> \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:<hash> \
  --aggregation <FedAvg|EntropyWeightedAvg|...> \
  --topology <RandomTopology|AffinityTopology|...> \
  --sign-with creator.key --output swarm.qgl

# Manifest doğrula
quinkgl manifest verify swarm.qgl --trusted-pubkey ed25519:<pubkey>

# Peer başlat
quinkgl run --manifest swarm.qgl --script peer_script.py \
  --node-id peer-1 --port 7001 \
  --trust-policy tofu \
  --script-arg data_root=./data

# Bilgi gör
quinkgl info
```

---

*Bu rehber QuinkGL v0.3.1 için yazılmıştır.*
