#!/bin/sh
chown root:root /home/exoria
chmod 755 /home/exoria
chown 1000:1000 /home/exoria/sessions || true
chmod 775 /home/exoria/sessions || true
chown 1000:1000 /home/exoria/sessions_envoyees || true
chmod 775 /home/exoria/sessions_envoyees || true

# La flotte (~30 machines) ouvre de nombreuses connexions SFTP en parallèle,
# bien au-delà du MaxStartups par défaut (10:30:100), ce qui fait dropper
# sshd des connexions en pré-auth ("drop connection ... Maxstartups").
if ! grep -q '^MaxStartups' /etc/ssh/sshd_config; then
    echo "MaxStartups 100:30:200" >> /etc/ssh/sshd_config
fi

# Les clients (paramiko) abandonnent parfois une connexion en plein transfert
# (timeout, erreur réseau) sans la fermer proprement côté serveur : la session
# sshd-session reste en "sleeping" indéfiniment. Sur 30 machines qui ouvrent
# une connexion par fichier, ça s'accumule en quelques milliers de process
# zombies qui finissent par saturer le CPU/RAM du conteneur et ralentir tous
# les transferts en cours. ClientAlive permet à sshd de détecter ces sessions
# mortes et de les fermer après ~45s d'inactivité.
if ! grep -q '^ClientAliveInterval' /etc/ssh/sshd_config; then
    echo "ClientAliveInterval 15" >> /etc/ssh/sshd_config
    echo "ClientAliveCountMax 3" >> /etc/ssh/sshd_config
fi
