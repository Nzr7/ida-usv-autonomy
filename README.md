# IDA USV Autonomy

Gerçek zamanlı **YOLO** duba algılama + kaçınma, hat/koridor takibi ve **son WPT’de durup renk bekleme (COLOR WAIT)** özellikli otonom USV yazılımı.  
ArduPilot/SITL ile **MAVLink** (UDP 14551) üzerinden haberleşir, H264 **UDP video** (port 5600) akışını işler.

## Demo Görselleri
![HUD / Koridor görünümü](docs/media/surus.png)
![Kırmızı hedef algısı](docs/media/surus2.jpg)

---

## Özellikler (Özet)
- **Algılama:** YOLO + HSV/LAB renk füzyonu; sarı/turuncu/kırmızı/yeşil/siyah.
- **Kaçınma:** Koridor bandı, `avoid_deg`; **preempt / emergency / panic** tetikleri.
- **Çatışma & Stuck:** İki taraf dolu → **conflict**, yerinde sayma → **bypass (kama)**.
- **Hat takibi:** Stanley denetimi + **köşe kilidi**.
- **COLOR WAIT:** Son WPT’ye varınca **durup MAVLink’ten gelen rengi bekler**  
  (STATUSTEXT “COLOR: …”, NAMED_VALUE_INT/PARAM_VALUE `COLOR/RENK` 0..4).

---

## Kurulum
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


model = YOLO("/mutlak/yol/weights/best.pt")


source .venv/bin/activate
python src/ida_gorev.py
# Pencereden çıkış: q


ida-usv-autonomy/
├─ src/ida_gorev.py         # ana uygulama
├─ configs/
│  ├─ params_default.yaml
│  └─ waypoints_example.yaml
├─ docs/
│  ├─ IDA_Bilgilendirme_Dokumani_3KOhK-2.pdf
│  ├─ yeni.pdf
│  └─ media/
│     ├─ surus.png
│     └─ surus2.jpg
├─ scripts/run_sim.sh
├─ requirements.txt
└─ README.md



> **En kritik nokta:** Komut bitince *yeni satıra* `EOF` yazmayı unutma. Aksi halde dosya boş/bozuk kalır.

## 3) Dosyayı kontrol et
```bash
ls -lh README.md
wc -l README.md          # 40+ satır görmelisin
head -5 README.md


