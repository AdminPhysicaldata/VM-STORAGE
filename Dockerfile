FROM atmoz/sftp:alpine
COPY sftp-setup.sh /etc/sftp.d/fix-perms.sh
RUN chmod +x /etc/sftp.d/fix-perms.sh
