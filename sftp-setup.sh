#!/bin/sh
chown root:root /home/exoria
chmod 755 /home/exoria
chown 1000:1000 /home/exoria/sessions || true
chmod 775 /home/exoria/sessions || true

# La flotte (~30 machines) ouvre de nombreuses connexions SFTP en parallèle,
# bien au-delà du MaxStartups par défaut (10:30:100), ce qui fait dropper
# sshd des connexions en pré-auth ("drop connection ... Maxstartups").
if ! grep -q '^MaxStartups' /etc/ssh/sshd_config; then
    echo "MaxStartups 100:30:200" >> /etc/ssh/sshd_config
fi
