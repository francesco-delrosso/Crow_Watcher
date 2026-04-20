import os
import sys

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
import queue
from config import TELEGRAM_TOKEN

# ============================================================
# IMPOSTAZIONI
# ============================================================

STREAM_URL              = "http://admin:admin@192.168.1.56:8081/video"
CARTELLA_VIDEO          = "/sdcard/rilevatore_corvi/"
MODELLO_AI              = "/sdcard/rilevatore_corvi/yolov8n.onnx"
DATABASE                = "/sdcard/rilevatore_corvi/corvi.db"
SOGLIA_CONFIDENZA       = 0.25
CLASSE_UCCELLO          = 14
DIMENSIONE_MODELLO      = 320
SECONDI_DOPO_ULTIMO     = 15      # secondi di registrazione dopo ultimo rilevamento
RISOLUZIONE_SALVATAGGIO = (1280, 720)
FPS_SALVATAGGIO         = 15
RISOLUZIONE_TELEGRAM    = (854, 480)
FPS_TELEGRAM            = 15
TELEGRAM_CANALE         = "@crowwatcher"
FILE_UTENTI             = "/sdcard/rilevatore_corvi/utenti.txt"
FILE_OFFSET             = "/sdcard/rilevatore_corvi/telegram_offset.txt"

FRASI_CORVI = [
    "🧠 QI superiore alla media. Non tua, quella dei corvidi.",
    "🪞 Ha superato il test dello specchio. Molti influencer no.",
    "🎓 Ha un cervello grande rispetto al corpo quanto quello dei delfini. Anche lui non nuota.",
    "🧩 Risolve puzzle a più step. Ha scoperto le cause-effetto. Congratulazioni a lui.",
    "📦 Nasconde il cibo e ricorda centinaia di nascondigli. Funziona meglio di Google Maps.",
    "🔢 Conta fino a 8. Più di quante dita usi per scrivere i messaggi vocali.",
    "📐 Capisce i concetti di volume e peso. Ha fatto fisica senza laurearsi.",
    "🧪 Usa strumenti di pietra. Ha saltato il Neolitico direttamente al problem-solving.",
    "🎯 Pianifica le mosse in anticipo come un giocatore di scacchi. Con le ali.",
    "🤯 Il suo cervello in proporzione è tra i più grandi del regno animale. Non commentare.",
    "📚 Impara guardando gli altri sbagliare. Tecnica che molti umani non hanno ancora adottato.",
    "🔬 Capisce la causalità. Sa che se fa X succede Y. Nobel non ancora assegnato.",
    "👁️ Ti ha già visto. Catalogato. Ha già deciso cosa pensa di te.",
    "😤 Se lo offendi, aspetta. Mesi. Anni se necessario. Non ha fretta.",
    "🗂️ Tiene un archivio mentale dei nemici. Aggiornato in tempo reale.",
    "📸 Ricorda le facce meglio di te dopo tre Aperol Spritz.",
    "🎭 Non dimentica chi è stato gentile con lui. Lista corta, molto curata.",
    "🕰️ Ha aspettato 6 mesi per vendicarsi di uno studioso che lo aveva spaventato. Pazienza infinita.",
    "🧳 Ha una lista nera. Sei in una lista. Non sappiamo quale.",
    "💾 Memoria a lungo termine attiva. Considera cosa hai fatto negli ultimi anni.",
    "💀 Un gruppo di corvi si chiama 'omicidio'. Il nome lo hanno scelto loro.",
    "🤝 Si aiutano a vicenda anche senza essere parenti. Concetto rivoluzionario.",
    "😴 Dorme in gruppi enormi per sicurezza. Introvertito di giorno, animale sociale di notte.",
    "⚖️ Ha un senso della giustizia. Punisce chi imbroglia. Anche senza toga.",
    "🗣️ Tiene discorsi agli altri corvi. Non sappiamo cosa dice. Probabilmente niente di buono.",
    "👥 Fa il funerale ai corvi morti. Raduno solenne, nessuna battuta.",
    "🎪 Gioca con altri animali per divertimento. Ha un concetto di svago più sviluppato di molti.",
    "📣 Avvisa gli altri del pericolo anche a costo suo. Solidarietà vera.",
    "🚨 Organizza mob collettivi contro i predatori. Sa fare rete.",
    "🎁 Porta regali alle persone che gli piacciono. Rifletti sul fatto che a te non ha portato niente.",
    "🚗 Usa le macchine per schiacciare le noci. Ha capito come sfruttare gli umani.",
    "🕵️ Usa ramoscelli come strumenti. Ha inventato il bastone prima di noi.",
    "🪝 Piega i fili per farne uncini. Brevetto non depositato.",
    "🌊 Sa usare l'acqua per ammorbidire il cibo duro. Chef non certificato.",
    "✂️ Taglia le foglie nella forma giusta per i suoi scopi. Artigiano autodidatta.",
    "🔑 Apre contenitori con coperchi a vite. Considera di cambiare le serrature.",
    "🪨 Lascia cadere sassi sugli intrusi dall'alto. Ingegneria difensiva.",
    "🎣 Usa il pane come esca per pescare. Ha inventato la pesca sportiva.",
    "🏗️ Costruisce nidi strutturati con strati diversi per isolamento. Geometra abusivo.",
    "🌡️ Sceglie materiali diversi in base alla temperatura. Edilizia sostenibile.",
    "🗣️ Dialetti diversi tra corvi di zone diverse. Più cultura linguistica del turista medio.",
    "📡 Oltre 250 vocalizzi distinti. Ha un vocabolario più ricco di certi tiktoker.",
    "🤫 Comunicazione silenziosa con gli occhi tra partner. Coppia affiatata.",
    "📢 Sa imitare la voce umana. Probabilmente lo sta già facendo.",
    "🔊 Sa imitare altri animali per ingannarli. Doppiogiochista certificato.",
    "💬 Ha un linguaggio specifico per 'pericolo', 'cibo', 'vieni qui'. Tre parole essenziali.",
    "🎵 Canta per piacere, non solo per comunicare. Ha già un album in mente.",
    "📻 Trasmette informazioni culturali alle generazioni successive. Tradizione orale funzionante.",
    "🌍 Vive su ogni continente tranne l'Antartide. Ha fatto scelte migliori di molti.",
    "🏙️ Prospera nelle città meglio della maggior parte dei residenti.",
    "🗑️ Ha imparato a vivere di spazzatura umana. Economia circolare.",
    "🌧️ Attivo in qualsiasi condizione meteo. Zero giorni di malattia.",
    "🌿 Onnivoro totale. Nessuna intolleranza alimentare.",
    "🏃 Velocità di volo fino a 50 km/h. Arriva prima di te ovunque.",
    "🌱 Si adatta a qualsiasi habitat in poche generazioni. Startup del mondo animale.",
    "🔄 Cambia strategia se quella attuale non funziona. Agilità mentale reale.",
    "🛡️ Sfrutta gli umani come protezione involontaria. Parassitismo elegante.",
    "🌐 Colonizzato ogni nicchia disponibile sul pianeta. Espansionismo senza eserciti.",
    "💍 Si accoppia per tutta la vita. Fedeltà: 100%. Commento: omesso.",
    "👶 Insegna ai piccoli cosa fare e cosa non fare. Sistema educativo: funzionante.",
    "🏠 Torna allo stesso nido per anni. Ha risolto il problema degli affitti.",
    "👫 Il maschio porta cibo alla femmina durante la cova. Romantico o strategico, non si sa.",
    "🧸 I piccoli restano con i genitori fino a 3 anni. Almeno loro se ne vanno.",
    "🎓 I giovani vanno a scuola dai vecchi. Sistema di apprendistato medievale ma funziona.",
    "❤️ Fa le coccole alla compagna. Sì, i corvi fanno le coccole.",
    "🪺 Decora il nido con oggetti brillanti. Interior design istintivo.",
    "🐦‍⬛ È venuto, ha visto, ha giudicato silenziosamente, se n'è andato.",
    "🎭 Può fare il morto per ingannare i predatori. Tecnica usata anche in molte riunioni.",
    "☃️ Gioca nella neve per puro divertimento. Ha più work-life balance di qualsiasi CEO.",
    "🔮 Associato a presagi in ogni cultura del mondo. Coincidenza? Lui non commenta.",
    "⚡ Nella mitologia nordica era il messaggero di Odino. Carriera pregressa notevole.",
    "🌑 Simbolo di morte in Occidente, fortuna in Oriente. Ha un'immagine ambivalente.",
    "🪄 Considerato magico per migliaia di anni. PR naturale senza ufficio stampa.",
    "📖 Citato in Edgar Allan Poe, Shakespeare, Esopo. Curriculum letterario invidiabile.",
    "🌀 Associato alla trasformazione in quasi tutte le culture. Coach motivazionale ante litteram.",
    "🏺 Raffigurato nelle pitture rupestri. Presenza mediatica da 40.000 anni.",
    "📷 Sa che lo stai guardando. Ti sta valutando.",
    "🤔 Non sta mangiando. Sta pensando. Differenza sottile ma importante.",
    "😑 Quella faccia che fa? Non è neutra. Sta giudicando.",
    "🎬 Ogni suo movimento è intenzionale. Non esiste il 'per caso' nel vocabolario corvino.",
    "🕶️ Aspetto minaccioso, comportamento diplomatico. Estetica dark, valori solidi.",
    "🌟 Non ha bisogno della tua approvazione. Non l'ha mai cercata.",
    "🧘 Non si stresa. Osserva. Aspetta. Agisce al momento giusto.",
    "💼 Se fosse umano, sarebbe il tipo silenzioso in ufficio che poi diventa il capo.",
    "🎲 Non lascia niente al caso. Ha già calcolato le probabilità.",
    "🚀 Esiste da 17 milioni di anni. Noi da 300.000. Chi è il nuovo arrivato?",
    "👑 Non chiede rispetto. Lo ottiene automaticamente.",
    "🔭 Osserva dall'alto. Letteralmente e metaforicamente.",
    "⏳ Ha tutto il tempo del mondo. Non ha un calendario da rispettare.",
    "🎩 Elegante, intelligente, imprevedibile. Il James Bond degli uccelli.",
    "🌙 Attivo anche di notte se necessario. Non conosce il concetto di orario.",
    "🏆 Nessun predatore naturale da adulto. Ha raggiunto la vetta della catena alimentare locale.",
    "💭 Cosa pensa? Non lo sappiamo. Lui sì.",
    "🎯 Ogni azione ha uno scopo. Lo spreco di energia non è nel suo vocabolario.",
    "🦾 Sopravvissuto a cinque estinzioni di massa. Hai presente la resilienza? Ecco.",
    "🌈 Piumaggio che riflette colori cangianti sotto la luce. Bello e non lo sa. O lo sa.",
    "🍷 Se fosse un vino sarebbe un Barolo del 2008. Complesso, austero, incompreso dai più.",
    "🎻 Nella cultura giapponese porta fortuna. In quella irlandese porta sfortuna. Lui se ne frega in entrambi i casi.",
    "🥷 Silenzioso quando vuole. Quando lo senti è perché ha deciso di farti sapere che c'è.",
    "🌊 Ha attraversato oceani in migrazioni da 3.000 km. Il viaggio in ritardo alla stazione ti sembra ancora un problema?",
]


# ============================================================
# THREAD CAMERA — legge frame in continuo
# ============================================================

_camera_frame  = None
_camera_lock   = threading.Lock()
_camera_attiva = True

def _camera_worker(cap):
    global _camera_frame
    while _camera_attiva:
        ok, f = cap.read()
        if ok and f is not None:
            with _camera_lock:
                _camera_frame = f

def leggi_frame():
    with _camera_lock:
        return _camera_frame


# ============================================================
# THREAD AI — analizza frame con tiling 3x2
# ============================================================

_coda_ai = queue.Queue(maxsize=1)
_conf_ai = 0.0
_lock_ai = threading.Lock()

def _ai_worker(rete):
    global _conf_ai
    while True:
        try:
            frame = _coda_ai.get(timeout=1)
        except queue.Empty:
            continue
        conf = _analizza_frame(rete, frame)
        with _lock_ai:
            _conf_ai = conf

def _analizza_frame(rete, frame):
    miglior = 0.0
    debug = []

    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (DIMENSIONE_MODELLO, DIMENSIONE_MODELLO),
        mean=(0,0,0), swapRB=True, crop=False
    )
    rete.setInput(blob)
    pred = np.squeeze(rete.forward()).T
    for det in pred:
        ps = det[4:]
        cls = int(np.argmax(ps))
        conf = float(ps[cls])
        if conf > 0.04:
            debug.append((conf, cls))
        if cls == CLASSE_UCCELLO and conf > miglior:
            miglior = conf

    nomi = {0:'persona', 2:'auto', 14:'uccello', 15:'gatto', 16:'cane'}
    debug.sort(reverse=True)
    top = [f"{'🐦' if c==14 else ''}{nomi.get(c,f'cls{c}')} {v*100:.0f}%"
           for v, c in debug[:3]]
    tag = '🐦 CORVO!' if miglior >= SOGLIA_CONFIDENZA else ''
    print(f"\r[AI] {tag} {' | '.join(top) if top else '-'}          ", end='', flush=True)
    return miglior

def leggi_conf():
    with _lock_ai:
        return _conf_ai

def manda_frame_ai(frame):
    try:
        _coda_ai.put_nowait(frame.copy())
    except queue.Full:
        pass


# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute("""CREATE TABLE IF NOT EXISTS avvistamenti (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_inizio TEXT NOT NULL,
        timestamp_fine TEXT,
        durata_secondi REAL,
        nome_video TEXT
    )""")
    conn.commit()
    conn.close()

def salva_avvistamento(t_inizio, t_fine, durata, video):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "INSERT INTO avvistamenti (timestamp_inizio,timestamp_fine,durata_secondi,nome_video) VALUES (?,?,?,?)",
            (t_inizio, t_fine, durata, video))
        conn.commit()
        conn.close()
        print(f"\n[DB] Salvato: {durata:.0f}s")
    except Exception as e:
        print(f"\n[DB] Errore: {e}")

def conta_oggi():
    try:
        oggi = time.strftime("%Y-%m-%d")
        conn = sqlite3.connect(DATABASE)
        n = conn.execute("SELECT COUNT(*) FROM avvistamenti WHERE timestamp_inizio LIKE ?",
                         (f"{oggi}%",)).fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# ============================================================
# TELEGRAM
# ============================================================

def leggi_utenti():
    if not os.path.exists(FILE_UTENTI):
        return []
    with open(FILE_UTENTI) as f:
        return [r.strip() for r in f if r.strip()]

def registra_nuovi_utenti():
    offset = 0
    if os.path.exists(FILE_OFFSET):
        with open(FILE_OFFSET) as f:
            c = f.read().strip()
            if c: offset = int(c)
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                         params={'offset': offset, 'timeout': 2}, timeout=5)
        if r.status_code != 200: return
        aggiornamenti = r.json().get('result', [])
        if not aggiornamenti: return
        esistenti = set(leggi_utenti())
        nuovi = []
        for a in aggiornamenti:
            nuovo_offset = a['update_id'] + 1
            if nuovo_offset > offset: offset = nuovo_offset
            cid = str(a.get('message',{}).get('chat',{}).get('id',''))
            if cid and cid not in esistenti:
                esistenti.add(cid)
                nuovi.append(cid)
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                              data={'chat_id': cid, 'text': "Registrato! Riceverai i video dei corvi."}, timeout=10)
        if nuovi:
            with open(FILE_UTENTI, 'a') as f:
                for u in nuovi: f.write(u + '\n')
        with open(FILE_OFFSET, 'w') as f:
            f.write(str(offset))
    except Exception:
        pass

LIMITE_TELEGRAM_MB = 49

def comprimi_video(path_orig):
    base, ext = os.path.splitext(path_orig)
    out = base + "_tg" + ext
    mb1 = os.path.getsize(path_orig) / 1024 / 1024
    for crf in [20, 25, 30, 35, 40]:
        try:
            subprocess.run([
                "ffmpeg", "-i", path_orig,
                "-vf", f"scale={RISOLUZIONE_TELEGRAM[0]}:-2",
                "-c:v", "libx264", "-crf", str(crf),
                "-preset", "fast", "-r", str(FPS_TELEGRAM),
                "-an", "-y", out
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        except Exception:
            return path_orig
        if os.path.exists(out):
            mb2 = os.path.getsize(out) / 1024 / 1024
            print(f"\n[FFMPEG] {mb1:.1f}MB → {mb2:.1f}MB (CRF {crf})")
            if mb2 <= LIMITE_TELEGRAM_MB:
                return out
    return out

def invia_video_telegram(path_video, durata_sec):
    path_invio = comprimi_video(path_video)
    n = conta_oggi()
    ora = time.strftime("%H:%M")
    fatto = random.choice(FRASI_CORVI)
    emoji_n = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    em = emoji_n[min(n-1, 9)] if n >= 1 else "🐦‍⬛"
    testo = (f"🐦‍⬛ CORVO AVVISTATO 🐦‍⬛\n\n"
             f"🕐 Ore {ora}\n"
             f"⏱️ Durata clip: {durata_sec:.0f} secondi\n"
             f"{em} Avvistamento n.{n} di oggi\n\n"
             f"{fatto}")
    try:
        with open(path_invio, 'rb') as f:
            contenuto = f.read()
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
            data={'chat_id': TELEGRAM_CANALE, 'caption': testo},
            files={'video': ('video.mp4', contenuto, 'video/mp4')},
            timeout=180)
        if r.status_code == 200:
            print(f"\n[TELEGRAM] Inviato a {TELEGRAM_CANALE}")
        else:
            print(f"\n[TELEGRAM] Errore: {r.status_code}")
    except Exception as e:
        print(f"\n[TELEGRAM] Errore: {e}")
    finally:
        if path_invio != path_video and os.path.exists(path_invio):
            os.remove(path_invio)

def aggiungi_compilazione(path_video):
    oggi = time.strftime("%Y%m%d")
    compilazione = os.path.join(CARTELLA_VIDEO, f"compilazione_{oggi}.mp4")
    lista = os.path.join(CARTELLA_VIDEO, f"lista_{oggi}.txt")
    with open(lista, 'a') as f:
        f.write(f"file '{path_video}'\n")
    try:
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", lista,
            "-c", "copy", "-y", compilazione
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(compilazione):
            mb = os.path.getsize(compilazione) / 1024 / 1024
            print(f"\n[COMPILAZIONE] → {mb:.1f}MB")
    except Exception as e:
        print(f"\n[COMPILAZIONE] Errore: {e}")

def invia_e_elimina(path_video, durata):
    invia_video_telegram(path_video, durata)
    try:
        os.remove(path_video)
    except Exception:
        pass


# ============================================================
# AVVIA REGISTRAZIONE FFMPEG
# ============================================================

def avvia_ffmpeg(nome_file):
    lw, lh = RISOLUZIONE_SALVATAGGIO
    return subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{lw}x{lh}", "-pix_fmt", "bgr24",
        "-r", str(FPS_SALVATAGGIO), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", nome_file
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def scrivi_frame(proc, frame):
    rid = cv2.resize(frame, RISOLUZIONE_SALVATAGGIO)
    try:
        proc.stdin.write(rid.tobytes())
    except Exception:
        pass

def chiudi_ffmpeg(proc):
    try:
        proc.stdin.close()
        proc.wait(timeout=30)
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("   RILEVATORE DI CORVI v2")
    print("=" * 50)

    os.makedirs(CARTELLA_VIDEO, exist_ok=True)
    init_db()

    if not os.path.exists(MODELLO_AI):
        print(f"ERRORE: modello non trovato: {MODELLO_AI}")
        return

    print("Caricamento modello AI...")
    rete = cv2.dnn.readNetFromONNX(MODELLO_AI)
    print("Modello OK")

    print(f"Connessione camera: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("ERRORE: camera non raggiungibile")
        return

    # Risoluzione gestita da USB Camera Pro, non serve settarla qui

    threading.Thread(target=_camera_worker, args=(cap,), daemon=True).start()
    while leggi_frame() is None:
        time.sleep(0.05)
    print("[CAM] OK")

    threading.Thread(target=_ai_worker, args=(rete,), daemon=True).start()
    print("[AI] OK")

    # ---- STATO ----
    registrando     = False
    ffmpeg_proc     = None
    file_video      = None
    ts_inizio       = None
    t_ultimo_corvo  = 0
    t_inizio_rec    = 0
    cnt             = 0
    t_ultimo_utenti = 0

    print("\n" + "=" * 50)
    print("  ATTIVO — CTRL+C per uscire")
    print("=" * 50 + "\n")

    dt = 1.0 / FPS_SALVATAGGIO
    prossimo = time.time()

    try:
        while True:
            now = time.time()
            wait = prossimo - now
            if wait > 0:
                time.sleep(wait)
            prossimo += dt

            frame = leggi_frame()
            if frame is None:
                continue

            cnt += 1
            now = time.time()

            # Manda frame ad AI ogni 5 frame
            if cnt % 15 == 0:
                manda_frame_ai(frame)

            # Leggi confidenza AI (nessun debounce, lettura diretta)
            conf = leggi_conf()
            corvo = conf >= SOGLIA_CONFIDENZA

            # Check utenti Telegram ogni 30s
            if now - t_ultimo_utenti >= 30:
                threading.Thread(target=registra_nuovi_utenti, daemon=True).start()
                t_ultimo_utenti = now

            # ---- LOGICA REGISTRAZIONE ----

            if corvo:
                t_ultimo_corvo = now

                if not registrando:
                    ts_inizio = time.strftime("%Y-%m-%d %H:%M:%S")
                    ts_file = time.strftime("%Y%m%d_%H%M%S")
                    file_video = os.path.join(CARTELLA_VIDEO, f"corvo_{ts_file}.mp4")
                    ffmpeg_proc = avvia_ffmpeg(file_video)
                    registrando = True
                    t_inizio_rec = now
                    print(f"\n[REC] Corvo! → {file_video}")

            if registrando:
                scrivi_frame(ffmpeg_proc, frame)
                durata = now - t_inizio_rec

                if corvo:
                    print(f"\r[REC] 🐦 {conf*100:.0f}% | {durata:.0f}s          ", end='', flush=True)
                else:
                    rimasti = SECONDI_DOPO_ULTIMO - (now - t_ultimo_corvo)
                    if rimasti > 0:
                        print(f"\r[REC] attesa... stop tra {rimasti:.0f}s | {durata:.0f}s          ", end='', flush=True)
                    else:
                        # STOP registrazione
                        chiudi_ffmpeg(ffmpeg_proc)
                        ffmpeg_proc = None
                        registrando = False
                        ts_fine = time.strftime("%Y-%m-%d %H:%M:%S")

                        print(f"\n[STOP] {durata:.0f}s registrati → {file_video}")
                        salva_avvistamento(ts_inizio, ts_fine, durata, file_video)
                        aggiungi_compilazione(file_video)
                        _fv = file_video
                        _d = durata
                        threading.Thread(target=invia_e_elimina, args=(_fv, _d), daemon=True).start()

                        file_video = None
                        ts_inizio = None
            else:
                if cnt % 30 == 0:
                    print(f"\r[IN ASCOLTO]                              ", end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nChiusura...")
    finally:
        if registrando and ffmpeg_proc:
            chiudi_ffmpeg(ffmpeg_proc)
            durata = time.time() - t_inizio_rec
            ts_fine = time.strftime("%Y-%m-%d %H:%M:%S")
            salva_avvistamento(ts_inizio, ts_fine, durata, file_video)
            aggiungi_compilazione(file_video)
            invia_e_elimina(file_video, durata)
        global _camera_attiva
        _camera_attiva = False
        cap.release()
        print("Terminato.")


if __name__ == "__main__":
    main()
