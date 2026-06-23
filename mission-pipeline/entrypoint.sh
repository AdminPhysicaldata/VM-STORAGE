#!/bin/bash
# Monte en SFTP les deux dossiers exposés par le serveur vm-storage
# ("sessions" et "sessions_envoyees", service "sftp" de son docker-compose),
# puis démarre pipeline_service.py dessus. Conçu pour tourner sur une machine
# distincte de vm-storage, qui n'a accès qu'au réseau (SFTP_HOST:SFTP_PORT).
#
# Doit être lancé avec --cap-add SYS_ADMIN --device /dev/fuse (FUSE en
# conteneur) ET --security-opt apparmor:unconfined sur un hôte Ubuntu/Debian :
# le profil AppArmor "docker-default" bloque l'appel système mount() même
# avec SYS_ADMIN, ce qui se traduit par "fuse: mount failed: Permission
# denied" — sans rapport avec les capabilities elles-mêmes.
#
# sshfs DOIT tourner avec -f (foreground) + un '&' shell pour le passer en
# arrière-plan : son fork interne (daemonize) est cassé dans Docker et échoue
# silencieusement (le point de montage s'enregistre mais toute opération
# filesystem renvoie une erreur I/O).
set -u

mkdir -p /data/sessions /data/session_envoye

mount_sftp() {
    local remote_dir="$1" local_dir="$2"
    echo "[mount] ${SFTP_USER}@${SFTP_HOST}:${SFTP_PORT}/${remote_dir} -> ${local_dir}" >&2
    echo "$SFTP_PASS" | sshfs \
        -p "$SFTP_PORT" \
        -o password_stdin \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o reconnect -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -f \
        "${SFTP_USER}@${SFTP_HOST}:${remote_dir}" "$local_dir" &
}

# Un dossier monté ou pas reste un dossier qui existe : stat()/ls() réussissent
# dans les deux cas (mkdir -p l'a créé localement avant le montage). Le seul
# test fiable est de vérifier dans /proc/mounts qu'un filesystem est
# effectivement monté à ce chemin précis.
is_mounted() {
    grep -qF " $1 fuse.sshfs " /proc/mounts
}

wait_for_mount() {
    local dir="$1"
    for _ in $(seq 1 30); do
        is_mounted "$dir" && return 0
        sleep 1
    done
    return 1
}

mount_sftp "sessions" /data/sessions
mount_sftp "sessions_envoyees" /data/session_envoye

ok=1
wait_for_mount /data/sessions       || { echo "[mount] /data/sessions non monté après 30s" >&2; ok=0; }
wait_for_mount /data/session_envoye || { echo "[mount] /data/session_envoye non monté après 30s" >&2; ok=0; }

if [ "$ok" -ne 1 ]; then
    echo "[mount] échec du montage SFTP — abandon. Vérifie que le conteneur tourne avec" >&2
    echo "[mount]   --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined" >&2
    exit 1
fi
echo "[mount] montages SFTP opérationnels (sessions, sessions_envoyees)" >&2

exec python3 -u pipeline_service.py
