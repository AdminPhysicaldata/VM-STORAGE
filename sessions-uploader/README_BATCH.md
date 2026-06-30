# Batch journalier poste 6 → validation → Mistral

Flux de bout en bout pour l'envoi quotidien d'un lot de ~5 Go du **poste 6**,
avec validation humaine sur le site et notification Web Push à 10h.

## Vue d'ensemble

```
03:00  build_daily_batch.py (vm-storage, cron)
       ├─ sélectionne poste 6, sessions de la veille, ≤5 Go (intégrité validée)
       ├─ copie le lot dans /data/session_envoye/batches/batch_<date>_rig_06  (copie SFTP de contrôle)
       └─ POST /api/pipeline/batches → statut pending_validation

10:00  backend (thread batch-notifier) → Web Push "X Go à valider" → /batch

 web   page /batch : sessions + taille + durée + chemin SFTP de la copie
       → bouton « Valider et envoyer » / « Rejeter »  (validation EN BLOC)
       → POST /api/pipeline/batches/<id>/validate → statut approved

*/10   send_validated_batches.py (vm-storage, cron)
       ├─ GET /api/pipeline/batches/approved-unsent
       └─ SessionsToMistral.py --only <manifeste> --all  (upload + suivi de run réutilisés)
            → mark-sent / mark-failed
```

## Composants

| Fichier | Rôle |
|---|---|
| `build_daily_batch.py` | Construit le lot du matin (sélection + copie SFTP + enregistrement) |
| `send_validated_batches.py` | Envoie les lots validés via `SessionsToMistral.py --only` |
| `SessionsToMistral.py` | Ajout de `--only MANIFEST` (envoi ciblé, sans filtre « jour courant ») |
| `cronjobs/crontab` | 03:00 build · `*/10` envoi |
| backend `routes/batches.py` | API des lots (`/api/pipeline/batches/*`) |
| backend `routes/push.py` | Web Push (abonnements + envoi + thread notif 10h) |
| front `pages/BatchPage.jsx` | Page `/batch` de validation |
| front `public/push-sw.js` | Handlers `push` / `notificationclick` du service worker |

## Déploiement

### 1. Générer les clés VAPID (une seule fois)

```bash
pip install py-vapid
python -m py_vapid --gen           # crée private_key.pem / public_key.pem
# Clé publique au format URL-safe base64 attendu par le navigateur :
python -m py_vapid --applicationServerKey   # -> VAPID_PUBLIC_KEY
```

Renseigner dans le `.env` de **vm-backend** :

```
VAPID_PUBLIC_KEY=BMx...           # applicationServerKey
VAPID_PRIVATE_KEY=...             # clé privée (contenu PEM ou base64url selon py_vapid)
VAPID_SUBJECT=mailto:chris.loisel94@gmail.com
BATCH_NOTIFY_HOUR=10
BATCH_NOTIFY_TZ=Indian/Antananarivo
```

`pywebpush` est déjà ajouté à `backend/requirements.txt` (rebuild de l'image backend nécessaire).

### 2. Activer le conteneur uploader (vm-storage)

Le service `sessions-uploader` est commenté dans `vm-storage/docker-compose.yml`.
Le réactiver avec les volumes et variables suivants :

```yaml
sessions-uploader:
  build: { context: ./sessions-uploader, dockerfile: Dockerfile }
  container_name: sessions-uploader
  restart: always
  volumes:
    - sessions_data:/data/sessions
    - sessions_envoye_data:/data/session_envoye
  environment:
    SESSIONS_DIR: /data/sessions
    BATCH_COPY_DIR: /data/session_envoye/batches
    BATCH_RIG_NUM: "6"
    BATCH_MAX_GB: "5"
    BACKEND_URL: http://${VM_BACKEND_IP}:5000/api
    INTERNAL_API_TOKEN: ${INTERNAL_API_TOKEN}
    TZ: Indian/Antananarivo            # pour que --date yesterday vise le bon jour
```

> `TZ` est important : `build_daily_batch.py` calcule « hier » à partir de l'heure
> locale du conteneur. Le poste tournant à Madagascar (UTC+3), aligner `TZ`.

### 3. Frontend

`npm run build` (déjà validé) régénère le service worker avec le handler push.
Le site doit être servi en **HTTPS** pour le Web Push.

### 4. iPhone — activer les notifications

1. Ouvrir le site dans Safari → Partager → **Ajouter à l'écran d'accueil**.
2. Ouvrir l'app depuis l'icône (mode PWA), aller sur **Batch du jour**.
3. Bouton **Activer les notifications** (iOS ≥ 16.4 requis).

## Tests rapides

```bash
# Construire un lot d'hier sans rien envoyer ni copier
python build_daily_batch.py --dry-run

# Construire pour une date précise
python build_daily_batch.py --date 2026-06-28 --max-gb 5

# Tester l'envoi d'un push à tous les abonnés (backend)
curl -X POST https://<site>/api/push/test -H "Authorization: Bearer <jwt>" \
     -H "Content-Type: application/json" -d '{"body":"coucou"}'

# Forcer l'envoi des lots validés
python send_validated_batches.py
```

## Statuts d'un lot

`pending_validation` → `approved` → `sending` → `sent`
                    ↘ `rejected`            ↘ `send_failed`
