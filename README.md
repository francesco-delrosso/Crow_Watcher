# Crow Watcher

Programma Python per tablet Android che usa la fotocamera per rilevare corvi (o uccelli in generale) tramite intelligenza artificiale. Quando rileva un corvo, inizia a registrare un video e lo invia automaticamente su Telegram.

---

## Come funziona

1. La fotocamera riprende continuamente tramite l'app **IP Webcam**
2. Il modello AI **YOLOv8n** analizza i fotogrammi cercando uccelli (classe 14 del dataset COCO)
3. Quando trova un corvo, inizia a registrare
4. Se il corvo esce dall'inquadratura, parte un timer di **30 secondi**
5. Se il corvo torna prima dei 30 secondi, il timer si azzera e la registrazione continua
6. Allo scadere dei 30 secondi, il video viene salvato **solo se** il corvo è stato visibile almeno **5 secondi** in totale
7. Il video viene compresso con **ffmpeg** e inviato a tutti gli utenti Telegram registrati

---

## Requisiti

### App da installare sul tablet Android

| App | Fonte | Funzione |
|-----|-------|----------|
| **Termux** | F-Droid (NON Play Store) | esegue Python |
| **IP Webcam** | Play Store | streaming fotocamera via HTTP |

### Librerie Python (installare in Termux)

```bash
pkg install python python-numpy opencv-python ffmpeg git
pip install requests
```

### File del modello AI

Il file `yolov8n.onnx` NON è incluso nel repo perché troppo grande. Generalo sul PC:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

Poi copia `yolov8n.onnx` in `/sdcard/rilevatore_corvi/` sul tablet.

---

## Installazione sul tablet

### Prima volta

```bash
termux-setup-storage
mkdir /sdcard/rilevatore_corvi
cd /sdcard/rilevatore_corvi
git clone https://github.com/francesco-delrosso/Crow_Watcher .
```

Crea il file `config.py` con il tuo token Telegram:

```bash
nano /sdcard/rilevatore_corvi/config.py
```

Scrivi dentro:

```python
TELEGRAM_TOKEN = "IL_TUO_TOKEN_QUI"
```

Salva con **CTRL+X → Y → Invio**.

### Aggiornare il codice

```bash
cd /sdcard/rilevatore_corvi && git pull
```

---

## Avvio

1. Apri **IP Webcam** → scorri in fondo → tocca **Avvia server**
2. In Termux:

```bash
termux-wake-lock
cd /sdcard/rilevatore_corvi
python corvi.py
```

---

## Struttura cartella sul tablet

```
/sdcard/rilevatore_corvi/
├── corvi.py              # script principale
├── config.py             # token Telegram (NON su GitHub)
├── yolov8n.onnx          # modello AI (NON su GitHub)
├── utenti.txt            # chat ID degli utenti registrati (auto-generato)
└── telegram_offset.txt   # offset aggiornamenti Telegram (auto-generato)

/sdcard/Videos/Corvi/     # qui vengono salvati i video
```

---

## Impostazioni principali

Tutte modificabili in cima a `corvi.py`:

| Impostazione | Valore attuale | Descrizione |
|---|---|---|
| `SECONDI_SENZA_CORVO` | 30 | secondi di assenza prima di fermare la registrazione |
| `SECONDI_MINIMI_CORVO` | 5 | secondi minimi di presenza per salvare il video |
| `SOGLIA_CONFIDENZA` | 0.4 | certezza minima AI (0.0 - 1.0) |
| `ANALIZZA_OGNI_N_FRAME` | 15 | ogni quanti frame analizzare con l'AI |
| `RISOLUZIONE_SALVATAGGIO` | 1280x720 | risoluzione video sul tablet |
| `FPS_SALVATAGGIO` | 60 | FPS video sul tablet |
| `RISOLUZIONE_TELEGRAM` | 854x480 | risoluzione video inviato su Telegram |
| `FPS_TELEGRAM` | 60 | FPS video inviato su Telegram |

---

## Bot Telegram

Il bot invia automaticamente i video a chiunque gli scriva un messaggio.

### Come ricevere i video

1. Cerca il bot su Telegram: **@Crow_Watcher_bot**
2. Scrivi qualsiasi messaggio
3. Ricevi conferma di registrazione
4. Da quel momento ricevi tutti i video dei corvi

### Creare il proprio bot (per chi clona il repo)

1. Scrivi a **@BotFather** su Telegram
2. Invia `/newbot` e segui le istruzioni
3. Copia il token ricevuto in `config.py`

---

## Peso video stimato

| | Tablet (salvato) | Telegram (inviato) |
|---|---|---|
| Risoluzione | 1280x720 | 854x480 |
| FPS | 60 | 60 |
| Peso/minuto | ~25 MB | ~6 MB |
| Limite 50MB Telegram | ~2 min | ~8 min |

---

## Problemi comuni

| Problema | Soluzione |
|----------|-----------|
| `No module named 'cv2'` | `pkg install opencv-python` |
| `ModuleNotFoundError: config` | crea `/sdcard/rilevatore_corvi/config.py` |
| Fotocamera non si connette | apri IP Webcam e tocca "Avvia server" |
| Modello non trovato | copia `yolov8n.onnx` in `/sdcard/rilevatore_corvi/` |
| Android sospende Termux | esegui `termux-wake-lock` prima di avviare |
| Video è una immagine statica | il tablet è troppo lento, aumenta `ANALIZZA_OGNI_N_FRAME` |
| Telegram non riceve | scrivi un messaggio al bot per registrarti |
| ffmpeg non trovato | `pkg install ffmpeg` |
