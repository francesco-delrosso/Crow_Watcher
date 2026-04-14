import os
import sys

# Silenziamo stderr prima di tutto per eliminare gli errori ffmpeg
devnull = open(os.devnull, 'w')
os.dup2(devnull.fileno(), sys.stderr.fileno())

import cv2
import numpy as np
import time
import sqlite3
import random
import requests
import subprocess
import threading
from config import TELEGRAM_TOKEN

# ============================================================
# IMPOSTAZIONI
# ============================================================

SECONDI_SENZA_CORVO     = 30
CARTELLA_VIDEO          = "/sdcard/rilevatore_corvi/"
MODELLO_AI              = "/sdcard/rilevatore_corvi/yolov8n.onnx"
DATABASE                = "/sdcard/rilevatore_corvi/corvi.db"
SOGLIA_CONFIDENZA       = 0.25
CLASSE_UCCELLO          = 14
DIMENSIONE_MODELLO      = 640
ZONA_RILEVAMENTO        = 0.70   # analizza solo il top X% del frame (es. 0.70 = ignora il 30% in basso)
ANALIZZA_OGNI_N_FRAME   = 10
SECONDI_MINIMI_CORVO    = 5
SECONDI_BUFFER_FINE     = 4    # secondi di buffer dopo che il corvo sparisce
RISOLUZIONE_SALVATAGGIO = (1280, 720)
FPS_SALVATAGGIO         = 30
RISOLUZIONE_TELEGRAM    = (854, 480)
FPS_TELEGRAM            = 30
TELEGRAM_CANALE         = "@crowwatcher"
FILE_UTENTI             = "/sdcard/rilevatore_corvi/utenti.txt"
FILE_OFFSET             = "/sdcard/rilevatore_corvi/telegram_offset.txt"
DEBUG_AI                = True

# Frasi casuali sui corvi per il messaggio Telegram
FRASI_CORVI = [
    "🧠 IQ stimato: più alto del tuo vicino di casa.",
    "👁️ Ti sta guardando. Ti ha già catalogato. Sa dove abiti.",
    "🎁 Porta doni alle persone che gli piacciono. Tu non hai ancora ricevuto niente.",
    "🚗 Usa le macchine per schiacciare le noci. Fondamentalmente guida meglio di molti.",
    "🗣️ Può imitare la voce umana. Probabilmente lo sta già facendo.",
    "☃️ Gioca nella neve per puro divertimento. Ha più work-life balance di te.",
    "💀 Un gruppo di corvi si chiama 'omicidio'. Nomen omen.",
    "🧳 Se lo tratti male, se lo ricorda. Per anni. Pianifica la vendetta con calma.",
    "🔧 Usa strumenti, risolve puzzle, apre contenitori. Candidato ideale per molti lavori.",
    "📡 Comunica con oltre 250 vocalizzi. Più di quanto tu faccia con i tuoi coinquilini.",
    "👶 Riconosce i volti umani meglio di alcuni sistemi di sorveglianza.",
    "🏆 Ha superato i test cognitivi progettati per i bambini di 7 anni. In meno tempo.",
]

# ============================================================
# DATABASE
# ============================================================

def inizializza_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS avvistamenti (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_inizio TEXT NOT NULL,
            timestamp_fine   TEXT,
            durata_secondi   REAL,
            nome_video       TEXT
        )
    """)
    conn.commit()
    conn.close()


def salva_avvistamento(timestamp_inizio, timestamp_fine, durata, nome_video):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "INSERT INTO avvistamenti (timestamp_inizio, timestamp_fine, durata_secondi, nome_video) VALUES (?,?,?,?)",
            (timestamp_inizio, timestamp_fine, durata, nome_video)
        )
        conn.commit()
        conn.close()
        print(f"[DB] Avvistamento salvato: {durata:.0f}s")
    except Exception as e:
        print(f"[DB] Errore: {e}")


def conta_avvistamenti_oggi():
    try:
        oggi = time.strftime("%Y-%m-%d")
        conn = sqlite3.connect(DATABASE)
        n = conn.execute(
            "SELECT COUNT(*) FROM avvistamenti WHERE timestamp_inizio LIKE ?",
            (f"{oggi}%",)
        ).fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# ============================================================
# AI
# ============================================================

def trova_uccelli(rete_ai, frame, larghezza_frame, altezza_frame):
    # Ritaglia la parte bassa del frame (strada) prima di analizzare
    righe_analisi = int(altezza_frame * ZONA_RILEVAMENTO)
    frame_analisi = frame[:righe_analisi, :]

    blob = cv2.dnn.blobFromImage(
        frame_analisi,
        scalefactor=1.0 / 255.0,
        size=(DIMENSIONE_MODELLO, DIMENSIONE_MODELLO),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    rete_ai.setInput(blob)
    predizioni = np.squeeze(rete_ai.forward()).T

    scala_x = larghezza_frame / DIMENSIONE_MODELLO
    scala_y = altezza_frame / DIMENSIONE_MODELLO
    uccelli_trovati = []
    miglior = (0.0, -1)

    for r in predizioni:
        punteggi = r[4:]
        classe   = int(np.argmax(punteggi))
        conf     = float(punteggi[classe])
        if conf > miglior[0]:
            miglior = (conf, classe)
        if classe == CLASSE_UCCELLO and conf >= SOGLIA_CONFIDENZA:
            cx = r[0] * scala_x
            cy = r[1] * scala_y
            w  = r[2] * scala_x
            h  = r[3] * scala_y
            uccelli_trovati.append({
                'x1': int(cx - w/2), 'y1': int(cy - h/2),
                'x2': int(cx + w/2), 'y2': int(cy + h/2),
                'confidenza': conf
            })

    if DEBUG_AI and miglior[0] > 0.05:
        nomi = {0:'persona', 14:'uccello', 15:'gatto', 16:'cane', 2:'auto'}
        nome = nomi.get(miglior[1], f'classe_{miglior[1]}')
        tag  = " <<< UCCELLO!" if miglior[1] == CLASSE_UCCELLO else ""
        print(f"[AI] {nome} {miglior[0]*100:.1f}%{tag}   ", end='\r')

    return uccelli_trovati


# ============================================================
# TELEGRAM
# ============================================================

def leggi_utenti():
    if not os.path.exists(FILE_UTENTI):
        return []
    with open(FILE_UTENTI, 'r') as f:
        return [r.strip() for r in f if r.strip()]


def registra_nuovi_utenti():
    offset = 0
    if os.path.exists(FILE_OFFSET):
        with open(FILE_OFFSET, 'r') as f:
            c = f.read().strip()
            if c:
                offset = int(c)
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={'offset': offset, 'timeout': 2}, timeout=5
        )
        if r.status_code != 200:
            return
        aggiornamenti = r.json().get('result', [])
        if not aggiornamenti:
            return
        esistenti = set(leggi_utenti())
        nuovi = []
        for a in aggiornamenti:
            nuovo_offset = a['update_id'] + 1
            if nuovo_offset > offset:
                offset = nuovo_offset
            chat_id = str(a.get('message', {}).get('chat', {}).get('id', ''))
            if chat_id and chat_id not in esistenti:
                esistenti.add(chat_id)
                nuovi.append(chat_id)
                print(f"\n[TELEGRAM] Nuovo utente: {chat_id}")
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    data={'chat_id': chat_id, 'text': "Sei registrato! Riceverai i video dei corvi."},
                    timeout=10
                )
        if nuovi:
            with open(FILE_UTENTI, 'a') as f:
                for uid in nuovi:
                    f.write(uid + '\n')
        with open(FILE_OFFSET, 'w') as f:
            f.write(str(offset))
    except Exception:
        pass


def aggiungi_a_compilazione(percorso_video):
    """Appende il video alla compilazione giornaliera."""
    oggi = time.strftime("%Y%m%d")
    compilazione = os.path.join(CARTELLA_VIDEO, f"compilazione_{oggi}.mp4")
    lista_file   = os.path.join(CARTELLA_VIDEO, f"lista_{oggi}.txt")

    # Aggiunge il video alla lista
    with open(lista_file, 'a') as f:
        f.write(f"file '{percorso_video}'\n")

    # Ricrea la compilazione con tutti i video del giorno
    print("[COMPILAZIONE] Aggiorno compilazione giornaliera...")
    try:
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", lista_file,
            "-c", "copy", "-y", compilazione
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(compilazione):
            mb = os.path.getsize(compilazione) / 1024 / 1024
            n  = open(lista_file).read().count("file '")
            print(f"[COMPILAZIONE] {n} video → {mb:.1f}MB → {compilazione}")
    except Exception as e:
        print(f"[COMPILAZIONE] Errore: {e}")


def invia_e_elimina_clip(percorso_video, secondi, ts):
    """Invia la clip su Telegram, poi la elimina dal tablet."""
    invia_video_telegram(percorso_video, secondi, ts)
    try:
        os.remove(percorso_video)
        print(f"[CLIP] Eliminata dal tablet: {os.path.basename(percorso_video)}")
    except Exception as e:
        print(f"[CLIP] Errore eliminazione: {e}")


def comprimi_video(percorso_originale):
    base, ext = os.path.splitext(percorso_originale)
    out = base + "_tg" + ext
    lw, lh = RISOLUZIONE_TELEGRAM
    print("[FFMPEG] Compressione...")
    try:
        subprocess.run([
            "ffmpeg", "-i", percorso_originale,
            "-vf", f"scale={lw}:{lh}",
            "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-r", str(FPS_TELEGRAM),
            "-an", "-y", out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(out):
            mb1 = os.path.getsize(percorso_originale) / 1024 / 1024
            mb2 = os.path.getsize(out) / 1024 / 1024
            print(f"[FFMPEG] {mb1:.1f}MB → {mb2:.1f}MB")
            return out
    except Exception as e:
        print(f"[FFMPEG] Errore: {e}")
    return percorso_originale


def invia_video_telegram(percorso_video, secondi_visibile, timestamp_inizio):
    percorso_da_inviare = comprimi_video(percorso_video)
    utenti      = leggi_utenti()
    destinatari = [TELEGRAM_CANALE]
    print(f"[TELEGRAM] Invio al canale {TELEGRAM_CANALE}...")

    # Costruiamo il messaggio divertente
    n_oggi    = conta_avvistamenti_oggi()
    ora       = time.strftime("%H:%M", time.localtime())
    mb        = os.path.getsize(percorso_da_inviare) / 1024 / 1024
    fatto      = random.choice(FRASI_CORVI)

    emoji_numero = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    em = emoji_numero[min(n_oggi - 1, 9)] if n_oggi >= 1 else "🐦‍⬛"

    didascalia = (
        f"🐦‍⬛ CORVO AVVISTATO 🐦‍⬛\n"
        f"\n"
        f"🕐 Ore {ora}\n"
        f"⏱️ Visibile per {secondi_visibile:.0f} secondi\n"
        f"{em} Avvistamento numero {n_oggi} di oggi\n"
        f"\n"
        f"{fatto}"
    )

    try:
        with open(percorso_da_inviare, 'rb') as f:
            contenuto = f.read()
        for chat_id in destinatari:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
                    data={'chat_id': chat_id, 'caption': didascalia},
                    files={'video': ('video.mp4', contenuto, 'video/mp4')},
                    timeout=180
                )
                if r.status_code == 200:
                    print(f"[TELEGRAM] Inviato a {chat_id}")
                else:
                    print(f"[TELEGRAM] Errore {chat_id}: {r.status_code}")
            except Exception as e:
                print(f"[TELEGRAM] Errore {chat_id}: {e}")
    except Exception as e:
        print(f"[TELEGRAM] Errore: {e}")
    finally:
        if percorso_da_inviare != percorso_video and os.path.exists(percorso_da_inviare):
            os.remove(percorso_da_inviare)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("   RILEVATORE DI CORVI")
    print("=" * 55)

    if not os.path.exists(CARTELLA_VIDEO):
        os.makedirs(CARTELLA_VIDEO)

    inizializza_db()
    print("[DB] Database pronto")

    if not os.path.exists(MODELLO_AI):
        print(f"ERRORE: modello non trovato in {MODELLO_AI}")
        return

    print("Caricamento modello AI...")
    rete_ai = cv2.dnn.readNetFromONNX(MODELLO_AI)
    print("Modello caricato!")

    print("Connessione fotocamera...")
    fotocamera = cv2.VideoCapture("http://127.0.0.1:8080/video")
    if not fotocamera.isOpened():
        print("ERRORE: IP Webcam non raggiungibile.")
        return

    larghezza = int(fotocamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza   = int(fotocamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = int(fotocamera.get(cv2.CAP_PROP_FPS)) or FPS_SALVATAGGIO
    print(f"Fotocamera: {larghezza}x{altezza} @ {fps}fps")

    # --- Stato ---
    sta_registrando      = False
    scrittore_video      = None
    tempo_ultimo_corvo   = None
    nome_file_video      = None
    timestamp_inizio_av  = None   # timestamp inizio avvistamento per il DB
    contatore_frame      = 0
    uccelli_correnti     = []
    secondi_corvo_totali = 0.0
    ultimo_tick_corvo    = None
    ultimo_check_utenti  = 0

    print("\n" + "=" * 55)
    print("  ATTIVO — CTRL+C per uscire")
    print("=" * 55 + "\n")

    try:
        while True:
            ok, frame = fotocamera.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            contatore_frame += 1
            momento_attuale  = time.time()

            # Analisi AI ogni N frame
            if contatore_frame % ANALIZZA_OGNI_N_FRAME == 0:
                uccelli_correnti = trova_uccelli(rete_ai, frame, larghezza, altezza)

            # Check nuovi utenti ogni 30 secondi
            if momento_attuale - ultimo_check_utenti >= 30:
                threading.Thread(target=registra_nuovi_utenti, daemon=True).start()
                ultimo_check_utenti = momento_attuale

            corvo_visibile = len(uccelli_correnti) > 0

            if corvo_visibile:
                tempo_ultimo_corvo = momento_attuale
                if ultimo_tick_corvo is not None:
                    secondi_corvo_totali += momento_attuale - ultimo_tick_corvo
                ultimo_tick_corvo = momento_attuale

                if not sta_registrando:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    timestamp_inizio_av = time.strftime("%Y-%m-%d %H:%M:%S")
                    nome_file_video = os.path.join(CARTELLA_VIDEO, f"corvo_{ts}.mp4")
                    codec = cv2.VideoWriter_fourcc(*'mp4v')
                    scrittore_video = cv2.VideoWriter(
                        nome_file_video, codec, FPS_SALVATAGGIO, RISOLUZIONE_SALVATAGGIO
                    )
                    sta_registrando = True
                    print(f"\n[REC] Inizio → {nome_file_video}")

                scrittore_video.write(cv2.resize(frame, RISOLUZIONE_SALVATAGGIO))
                print(f"[REC] {len(uccelli_correnti)} corvo/i | {secondi_corvo_totali:.0f}s   ", end='\r')

            else:
                ultimo_tick_corvo = None

                if sta_registrando and tempo_ultimo_corvo is not None:
                    secondi_assenza = momento_attuale - tempo_ultimo_corvo

                    # Buffer di fine: scrivi ancora qualche secondo dopo che il corvo sparisce
                    if secondi_assenza < SECONDI_BUFFER_FINE:
                        scrittore_video.write(cv2.resize(frame, RISOLUZIONE_SALVATAGGIO))

                    secondi_rimasti = SECONDI_SENZA_CORVO - secondi_assenza
                    if secondi_rimasti > 0:
                        print(f"[TIMER] Stop tra {int(secondi_rimasti)}s | totale {secondi_corvo_totali:.0f}s   ", end='\r')

                    if secondi_assenza >= SECONDI_SENZA_CORVO:
                        scrittore_video.release()
                        scrittore_video = None
                        sta_registrando = False

                        timestamp_fine = time.strftime("%Y-%m-%d %H:%M:%S")

                        if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                            print(f"\n[SALVATO] {secondi_corvo_totali:.0f}s → {nome_file_video}")
                            # Salviamo nel DB
                            salva_avvistamento(
                                timestamp_inizio_av, timestamp_fine,
                                secondi_corvo_totali, nome_file_video
                            )
                            # Aggiunge alla compilazione giornaliera (sincrono, deve finire prima di eliminare)
                            aggiungi_a_compilazione(nome_file_video)
                            # Invia su Telegram e poi elimina la clip dal tablet
                            _ts = timestamp_inizio_av
                            _nf = nome_file_video
                            _s  = secondi_corvo_totali
                            threading.Thread(
                                target=invia_e_elimina_clip,
                                args=(_nf, _s, _ts),
                                daemon=True
                            ).start()
                        else:
                            os.remove(nome_file_video)
                            print(f"\n[SCARTATO] solo {secondi_corvo_totali:.0f}s di corvo")

                        nome_file_video      = None
                        tempo_ultimo_corvo   = None
                        uccelli_correnti     = []
                        secondi_corvo_totali = 0.0
                        timestamp_inizio_av  = None

                elif not sta_registrando:
                    print("[IN ASCOLTO]   ", end='\r')

    except KeyboardInterrupt:
        print("\n\nChiusura...")

    finally:
        if scrittore_video is not None:
            scrittore_video.release()
            if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                ts_fine = time.strftime("%Y-%m-%d %H:%M:%S")
                salva_avvistamento(timestamp_inizio_av, ts_fine, secondi_corvo_totali, nome_file_video)
                aggiungi_a_compilazione(nome_file_video)
                print(f"[SALVATO] {nome_file_video}")
                invia_e_elimina_clip(nome_file_video, secondi_corvo_totali, timestamp_inizio_av)
            else:
                os.remove(nome_file_video)
        fotocamera.release()
        print("Terminato.")


if __name__ == "__main__":
    main()
